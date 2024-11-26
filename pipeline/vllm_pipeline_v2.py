from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import random

from ultralytics import YOLO
import cv2
import copy
from pathlib import Path
from tqdm import tqdm
import requests
import json

url = "http://localhost:8001/v1/chat/completions"

headers = {
    "Content-Type": "application/json"
}
def chat(img_url='', query=''):
    if img_url != '':
        data = {
        "model": "Qwen2-VL-72B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": img_url}},
                {"type": "text", "text": query}
            ]}
        ] }
    else:
        data = {
        "model": "Qwen2-VL-72B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": query}
            ]}
        ] }
    return data

def resize(img_path, task):
    image = cv2.imread(img_path)
    resized_image = cv2.resize(image, None, fx=0.7, fy=0.7)
    cv2.imwrite(f'.../{task}_tmp.png', resized_image)


def _return_spf_qa(res_json, raw_path, idx=0):
    prompt = """Design some QA pairs based only on the icons in the picture, only on the text in the picture,
            only on some relationships between components and only on locations of components (such as the return icon is in the upper left corner of the screen, etc.), and give questions and correct answers."""
    resize(raw_path,task='spf_qa')
    data = chat('http://localhost:6666/spf_qa.png', prompt)
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response = response.json()
    response = response['choices'][0]['message']['content']

    prompt = "Please format the below data as JSON format such as {'question': ..., 'type': 'text' or 'icon' or 'relationship' or 'location', 'answer': ...}." + response
    data = chat('', prompt)
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response = response.json()
    raw_json = response['choices'][0]['message']['content']

    start, end = raw_json.find("["), raw_json.rfind("]")
    json_qa = ''
    if start == -1 or end == -1 or start == end:
        return
    else:
        try:
            json_qa = json.loads(raw_json[start:end+1])
            for item in json_qa:
                res_json.append({
                        "id":idx,
                        "img_path": raw_path,
                        "question": item['question'],
                        "type": item['type'],
                        "gt": item['answer']
                        })
        except :
            print('【error sample:】 ', json_qa)

    

def get_spf_qa(img_list):
    res_json = []
    output_path = 'Fusin_Gate_training/data_vllm/spf_qa'

    now = 0
    for i in tqdm(range(now, len(img_list))):
        img_path = img_list[i]
        _return_spf_qa(res_json, img_path, i)
        if i % 100 == 0:
            with open(f"{output_path}/spf_qa.json", 'w') as file:
                json.dump(res_json, file)

    with open(f"{output_path}/spf_qa.json", 'w') as file:
        json.dump(res_json, file)


def _return_screen(res_json, raw_path, idx=0):
    prompt = "Generate a summary of the screen in one sentence. Do not focus on specifically naming the various UI elements, but instead, focus on the content."
    resize(raw_path, 'Global_Des')
    data = chat('http://localhost:6666/Global_Des.png', prompt)
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response = response.json()
    

    question = "Describe this screen."
    if 'choices' in response:
        result = response['choices'][0]['message']['content']
    
        res_json.append({
            "id": idx,
            "img_path": raw_path,
            "question": question,
            "gt": result
        })
    else:
        print('# Error: ', response)


def get_global_des(img_list):
    res_json = []
    output_path = 'Fusion_Gate_training/data_vllm/global_des'
    now = 0
    for i in tqdm(range(now, len(img_list))):
        img_path = img_list[i]
        _return_screen(res_json, img_path, i)
        if i % 100 == 0:
            with open(f"{output_path}/Global_Des.json", 'w') as file:
                json.dump(res_json, file)

    with open(f"{output_path}/Global_Des.json", 'w') as file:
        json.dump(res_json, file)


def get_yolo_res(model, raw_path, output_path):
    img = cv2.imread(raw_path)
    result = model.predict(img, conf=0.2, save=False)[0]
    xyxy = result.boxes.xyxy.cpu().numpy()  # box with xyxy format, (N, 4)
    conf = result.boxes.conf.cpu().numpy()  # confidence score, (N, 1)
    label = result.boxes.cls.cpu().numpy()  # cls, (N, 1)

    #  bboxes_output= []
    bboxes = []
    file_name = Path(raw_path).stem
    for i, (a, b, c) in enumerate(zip(xyxy, conf, label)):
        if c != 8:
            continue
        # bboxxed_file = f'{file_name}_{i}.png'
        x0, y0, x1, y1 = a
        width, height = img.shape[1], img.shape[0]
        bboxes.append([x0/width, y0/height, x1/width, y1/height])
    return bboxes


def _return_local_des(res_json, bboxes, raw_path, idx=0):
    img = cv2.imread(raw_path)
    file_name = Path(raw_path).stem
    for i, bbox in enumerate(bboxes):
        img_copy = copy.deepcopy(img)
        x0, y0, x1, y1 = bbox
        width, height = img.shape[1], img.shape[0]
        cv2.rectangle(img_copy, (int(x0*width), int(y0*height)),
                      (int(x1*width), int(y1*height)), (255, 0, 0), 2)

        x0, y0, x1, y1 = int(1000*x0), int(1000*y0), int(1000*x1), int(1000*y1)
        coords_str = f'<|box_start|>({x0},{y0}),({x1},{y1})<|box_end|>'
        prompt = (
            "Describe this image. You will receive a screenshot of a GUI that includes a bounding box (bbox) "
            "with specified coordinates. Your task is to "
            "analyze the content within the bbox and identify the component to which it belongs by looking for "
            "surrounding component boundaries. Please provide a detailed description that includes the following:\n"
            "1. Identify the content inside the bbox (text or graphic element).\n"
            "2. Look for the component boundary surrounding the bbox and describe the overall component it belongs to.\n"
            "3. Explain the function of this component and any other relevant elements it contains.\n"
            "4. If there are no surrounding component boundaries, state that there are no related components nearby.\n"
            "**Output Example (response with just one sentence):**\n"
            "● \"This is an icon of a house, belonging to a button component that describes the home page; it also includes "
            "another house icon as part of this component.\"\n"
            "● \"This is an arrow icon, belonging to the 'General' row within the list, indicating that this is a clickable "
            "item in the menu which may goto the 'General' page.\"\n"
            "● \"This is a standalone button labeled 'Submit', and there are no related components nearby.\""
            "Now the coordinate of bbox i'd like you to analyse is " + coords_str
        )
        
        resized_image = cv2.resize(img_copy, None, fx=0.7, fy=0.7)
        cv2.imwrite('Fusion_Gate_training/data_vllm/local_des.png', resized_image)

        
        coords_str = f'[{x0},{y0},{x1},{y1}]'
        
        data = chat('http://localhost:6666/local_des.png', prompt)
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response = response.json()
        result = response['choices'][0]['message']['content']
        
        res_json.append({
                        "id":idx,
                        "img_path": raw_path,
                        "bbox": coords_str,
                        "gt": result
                        })


def get_local_des(img_list):
    yolo_model_path = "yolov8/save_weights/best.pt"
    yolo_model = YOLO(yolo_model_path).to('cuda')
    output_path = 'Fusion_Gate_training/data_vllm/local_des'
    res_json = []

    now = 0
    for i in tqdm(range(now, len(img_list))):
        img_path = img_list[i]
        bboxes = get_yolo_res(yolo_model, img_path,
                              'output_path/')
        _return_local_des(res_json, bboxes,
                          img_path, i)
        if i % 100 == 0:
            with open(f"{output_path}/local_des.json", 'w') as file:
                json.dump(res_json, file)

    with open(f"{output_path}/local_des.json", 'w') as file:
        json.dump(res_json, file)


img_txt = 'Fusion_Gate_training/data_vllm/img_path.txt'
with open(img_txt, 'r') as f:
    img_paths = [line.strip() for line in f.readlines()]


print("Generating Global Description.....")
get_global_des(img_paths)

print("Generating Local Description .....")
get_local_des(img_paths)

print("Generating QA .....")
get_spf_qa(img_paths)




