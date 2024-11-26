import json
import os
from PIL import Image
import random
from tqdm import tqdm
import random
import copy
from torchvision.ops import box_iou
import torch

def get_clean_data(json_path):
    size = get_resolution(json_path.replace('json', 'png'))
    result = None
    all_nodes = []
    with open(json_path, 'r') as f:
        data = json.load(f)['children']
        result = extract_components(data)
       
    def get_all_nodes(data):
        for node in data:
            extracted = {
            'componentLabel': node.get('componentLabel'),
            'bounds': node.get('bounds'),
            'iconClass': node.get('iconClass')}
            all_nodes.append(extracted)
            
            if 'children' in node:
                get_all_nodes(node['children'])
    
    get_all_nodes(result)  

    return result, all_nodes

def judge(bbox1, bbox2, size):
    a,b,c,d = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    A,B,C,D = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    w, h = size[0], size[1]
    for x in [a,b,c,d,A,B,C,D]:
        if x < 0:
            return True
    if a>w or A>w or c>w or C>w or b>h or B>h or d>h or D>h:
        return True
    return False

def get_resolution(image_path):
    with Image.open(image_path) as img:
        return img.size

def unify_bbox(bbox, size):
    w, h = size
    return [bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h]

def extract_components(data, parent=None):
    components = []
    for component in data:

        extracted = {
            'componentLabel': component.get('componentLabel'),
            'bounds': component.get('bounds'),
            'iconClass': component.get('iconClass')
        }

        if 'children' in component:
            extracted['children'] = extract_components(component['children'], extracted)
        components.append(extracted)
    return components

def compute_iou(bbox1, bbox2):
    bbox1 = torch.tensor([bbox1])
    bbox2 = torch.tensor([bbox2])
    ious = box_iou(bbox1, bbox2)
    return ious[0][0]

def get_negatives(node, parent, size, type_='4sibling', all_nodes=None, siblings=None, childrens=None):
    w, h = size[0], size[1]
    bbox1, bbox2 = node['bounds'], parent['bounds']
    s_left, s_top, s_right, s_low = bbox1[0], bbox1[1], bbox1[2], bbox1[3]

    if type_ == '4parent':
        max_count = 50
        count = 0
        while compute_iou([s_left, s_top, s_right, s_low], bbox1) > 0.1 or compute_iou([s_left, s_top, s_right, s_low], bbox2) > 0.3:
            s_left = random.randint(0, bbox1[0])
            s_top = random.randint(0, bbox1[1])
            s_right = random.randint(bbox1[2], w)
            s_low = random.randint(bbox1[3], h)
            count = count + 1
            if count > max_count:
                return None

        node_new = copy.deepcopy(parent)
        node_new['bounds'] = [s_left, s_top, s_right, s_low]
        return ((node, 3, node_new))

    else: # type_ == '4sibling'                    
        t = random.randint(0, len(all_nodes)-1)
        relevant_bboxs = [bbox2]
        for x in siblings:
            relevant_bboxs.append(x['bounds'])
        if childrens is not None:
            for x in childrens:
                relevant_bboxs.append(x['bounds'])
        if len(all_nodes) == len(relevant_bboxs):
            return None
        max_count = 50
        count = 0

        while all_nodes[t]['bounds'] in relevant_bboxs:   
            t = random.randint(0, len(all_nodes)-1)
            count = count + 1
            if count > max_count:
                return None
        
        return ((node, 4, all_nodes[t]))

def parse_json_hierarchy(data, size, num, all_nodes):

    result = []

    def dfs(node, parent=None):
        node_repr = {k: v for k, v in node.items() if k != 'children'}

        if parent is not None:
            parent_repr = {k: v for k, v in parent.items() if k != 'children'}
            result.append((parent_repr, 1, node_repr))
            result.append((node_repr, 2, parent_repr))
            for i in range(num):
                sample = get_negatives(node_repr, parent_repr, size, '4parent')
                if sample is not None:
                    result.append(sample) 

        if parent is not None and len(parent['children']) >= 2:
            siblings = parent['children']
            parent_repr = {k: v for k, v in parent.items() if k != 'children'}

            for sibling in siblings:
                if sibling is not node:
                    sibling_repr = {k: v for k, v in sibling.items() if k != 'children'}
                    result.append((node_repr, 0, sibling_repr, parent_repr))
            for i in range(num):
                sample = get_negatives(node_repr, parent_repr, size, '4sibling', all_nodes, siblings, node['children'] if 'children' in node else None)
                if sample is not None:
                    result.append(sample)
        
        if 'children' in node:
            for child in node['children']:
                if judge(node['bounds'], child['bounds'], size): continue
                dfs(child, node)
          

    for entry in data:
        dfs(entry)

    return result

def unif_bbox(bbox, size_source, scale):
    w, h = size_source[0], size_source[1]
    return [int((bbox[0] / w) * scale),
            int((bbox[1] / h) * scale),
            int((bbox[2] / w) * scale),
            int((bbox[3] / h) * scale) ]

def get_relation_prediction_data(data_path, save_path):
    train_pairs = []
    Uni_Labels_List = []
    part = 1
    a,b,c,d = 0,0,0,0
    pos, neg = 7000, 3000
    with open(data_path, 'r') as f:
        data = json.load(f)
        for img in tqdm(data):
            sem_json = 'rico_semantic/rico_dataset_v0.1_semantic_annotations/semantic_annotations/' + img.replace('jpg', 'json')
            if not os.path.exists(sem_json): continue

            clean_data, all_nodes = get_clean_data(sem_json)
            size = get_resolution('rico_semantic/rico_dataset_v0.1_semantic_annotations/semantic_annotations/' + img.replace('jpg', 'png'))

            output = parse_json_hierarchy(clean_data, size, 2, all_nodes) 
            
            for item in output:
                sample = {}
                sample['image_path'] = 'data_saving_fold/' + img
                scale = 1000
                node1_bbox = unif_bbox(item[0]['bounds'], size, scale)
                node1_label = 'Button' if item[0]['componentLabel'] == 'Text Button' else item[0]['componentLabel']
                node2_bbox = unif_bbox(item[2]['bounds'], size, scale)
                node2_label = 'Button' if item[2]['componentLabel'] == 'Text Button' else item[2]['componentLabel']
                node2_class = item[2]['iconClass']
                relation = item[1]

                if node1_label not in Uni_Labels_List:
                    Uni_Labels_List.append(node1_label)
                if node2_label not in Uni_Labels_List:
                    Uni_Labels_List.append(node2_label)
              
                if node1_bbox == node2_bbox: continue

                node1_type = node1_label if node1_label != '' else node1_class
                node2_type = node2_label if node2_label != '' else node2_class
                n1 = random.randint(0, len(QUESTION_PROMPT_FOR_RELATION_PREDICTION)-1)
                n2 = random.randint(0, len(SIBLING_RELATIONSHIP_ANSWER_PROMPT_FOR_RELATION_PREDICTION)-1)
                n3 = random.randint(0, len(CONTAINMENT_RELATIONSHIP_ANSWER_PROMPT_FOR_RELATION_PREDICTION)-1)
                n4 = random.randint(0, len(NO_CONTAINMENT_RELATIONSHIP_ANSWER_PROMPT_FOR_RELATION_PREDICTION)-1)
                n5 = random.randint(0, len(OTHERS)-1)

                question = QUESTION_PROMPT_FOR_RELATION_PREDICTION[n1].format(node1_bbox, node2_bbox)

                if relation == 0:
                    a += 1
                    parent_bbox = unif_bbox(item[-1]['bounds'], size, scale)
                    parent_type = 'Button' if item[-1]['componentLabel'] == 'Text Button' else item[-1]['componentLabel']
                    assert (parent_type != None) and (parent_type != '')
                    answer = SIBLING_RELATIONSHIP_ANSWER_PROMPT_FOR_RELATION_PREDICTION[n2].format(
                        bbox1=node1_bbox, type1=node1_type, bbox2=node2_bbox, type2=node2_type, bbox3=parent_bbox, type3=parent_type)

                elif relation == 1 or relation == 2:
                    b += 1
                    if relation == 1: node1_bbox, node1_type, node2_bbox, node2_type = node2_bbox, node2_type, node1_bbox, node1_type
                    answer = CONTAINMENT_RELATIONSHIP_ANSWER_PROMPT_FOR_RELATION_PREDICTION[n3].format(
                        bbox1=node1_bbox, type1=node1_type, bbox2=node2_bbox, type2=node2_type)
                
                elif relation == 3:
                    c += 1
                    answer = NO_CONTAINMENT_RELATIONSHIP_ANSWER_PROMPT_FOR_RELATION_PREDICTION[n4].format(node1_bbox, node2_bbox)

                elif relation == 4:
                    d += 1
                    answer = OTHERS[n5].format(bbox1=node1_bbox, type1=node1_type, bbox2=node2_bbox, type2=node2_type)

                sample['question'] = question
                sample['answer'] = answer

                train_pairs.append(sample)
       
                if len(train_pairs) == 100000:
                    random.shuffle(train_pairs)
                    print(f'### Saving {(part-1)*100000}-{part*100000} samples.......')
                    with open(save_path + f'{part}_' + 'relation_train.json', 'w') as f:
                        f.write(json.dumps(train_pairs))
                    train_pairs = []
                    part += 1 

    count = len(train_pairs)+ (part-1)*100000
    
    print(f'label=0:{a/count}, label=1,2:{b/count}, label=3:{c/count}, label=4:{b/count}')
    print(f'full num: {count}')
    print(f'{a}  {b}  {c}  {d}')
    print(Uni_Labels_List)

    random.shuffle(train_pairs)      
    with open(save_path+ f'{part}_' + 'relation_train.json', 'w') as f:
        f.write(json.dumps(train_pairs))



QUESTION_PROMPT_FOR_RELATION_PREDICTION = [
    "What is the relationship between the elements at {} and {}?",
    "How are the items situated in {} and {} related to one another?",
    "How do the elements in {} and {} relate to each other?",
    "There are two components at {} and {}. What is their relationship?",
    "What type of spatial relationship exists between the components at {} and {}?",
    "What is the connection between the item at {} and {}?"
]
#label=0
SIBLING_RELATIONSHIP_ANSWER_PROMPT_FOR_RELATION_PREDICTION = [
    "Both \"{type1}\" at {bbox1} and \"{type2}\" at {bbox2} are positioned inside \"{type3}\" at {bbox3}.",
    "\"{type1}\" at {bbox1} and \"{type2}\" at {bbox2} are siblings inside \"{type3}\" at {bbox3}.",
    "Both \"{type1}\" at {bbox1} and \"{type2}\" at {bbox2} are contained by \"{type3}\" at {bbox3}.",   
    "\"{type1}\" at {bbox1} and \"{type2}\" at {bbox2} are sibling elements from \"{type3}\" at {bbox3}."
]   
#label=1, 2
CONTAINMENT_RELATIONSHIP_ANSWER_PROMPT_FOR_RELATION_PREDICTION = [
    "The element at {bbox1} is \"{type1}\", contained in the \"{type2}\" at {bbox2}.",
    "At coordinates {bbox1}, the element is \"{type1}\", contained by the \"{type2}\" at {bbox2}.",
    "The item located at {bbox1} is identified as \"{type1}\", positioned inside the \"{type2}\" defined by {bbox2}.",
    "The region at {bbox2} is recognized as \"{type2}\", containing the component \"{type1}\" at {bbox1}.",
    "The area {bbox2} serves as \"{type2}\", including the element \"{type1}\" at {bbox1}.",
    "The space defined by {bbox2} is \"{type2}\" and contains the element \"{type1}\" at {bbox1}."
]
#label=3
NO_CONTAINMENT_RELATIONSHIP_ANSWER_PROMPT_FOR_RELATION_PREDICTION = [
    "The elements at coordinates {} and {} have no direct link.",
    "The components located at {} and {} do not have a direct connection.",
    "There is no direct relationship between the items at {} and {}.",
    "No direct relationship can be found between the components at {} and {}."
]
# no direct spatial relationship
OTHERS = [
    "\"{type1}\" at {bbox1} lacks a direct link with \"{type2}\" at {bbox2}.",
    "\"{type1}\" at {bbox1} and \"{type2}\" at {bbox2} are not directly related.",
    "\"{type1}\" at {bbox1} and \"{type2}\" at {bbox2} lack a direct relationship.",
    "There is no direct connection between \"{type1}\" at {bbox1} and \"{type2}\" at {bbox2}."
]

if __name__ == '__main__':
    save_path = 'SpatialRelationPrediction/'
    get_relation_prediction_data('dataset/rico_sem_train_index.json', save_path)

