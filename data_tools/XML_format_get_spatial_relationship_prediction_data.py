import json
import os
from PIL import Image
import random
from tqdm import tqdm
import random
import copy
from torchvision.ops import box_iou
import torch

def label_formulate(label: str) -> str:
    """
    Convert element types to a unified format.
    label: The original label of the element.
    """
    label = label.replace(' ', '_')
    if label == 'On/Off_Switch':
        label = 'Switch'
    if label == 'Text_Button':
        label = 'Button'
    return label


def get_clean_data(json_path: str) -> tuple:
    """
    Filter out items that do not meet the specifications.
    """
    size = get_resolution(json_path.replace('json', 'png'))
    result = None
    all_nodes = []
    with open(json_path, 'r') as f:
        data = json.load(f)['children']
        result = extract_components(data)

    def get_all_nodes(data: list) -> None:
        for node in data:
            extracted = {
                'componentLabel': label_formulate(node.get('componentLabel')),
                'bounds': node.get('bounds'),
                'iconClass': node.get('iconClass')}
            all_nodes.append(extracted)

            if 'children' in node:
                get_all_nodes(node['children'])

    get_all_nodes(result)

    return result, all_nodes


def judge(bbox1: list, bbox2: list, size: tuple) -> bool:
    """
    Determine whether bbox meets the specification
    """
    a, b, c, d = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    A, B, C, D = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    w, h = size[0], size[1]

    for x in [a, b, c, d, A, B, C, D]:
        if x < 0:
            return True
    if a > w or A > w or c > w or C > w or b > h or B > h or d > h or D > h:
        return True
    return False


def get_resolution(image_path: str) -> tuple:
    """
    Get the image resolution, i.e., width and height.
    """
    with Image.open(image_path) as img:
        return img.size


def unify_bbox(bbox: list, size: tuple) -> list:
    w, h = size
    return [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]


def extract_components(data: list, parent: dict = None) -> list:
    """
    Extracting attribute information of UI elements.
    """
    components = []

    for component in data:
        extracted = {
            'componentLabel': label_formulate(component.get('componentLabel')),
            'bounds': component.get('bounds'),
            'iconClass': component.get('iconClass')
        }

        if 'children' in component:
            extracted['children'] = extract_components(component['children'], extracted)
        components.append(extracted)
    return components


def compute_iou(bbox1: list, bbox2: list) -> float:
    bbox1 = torch.tensor([bbox1])
    bbox2 = torch.tensor([bbox2])
    ious = box_iou(bbox1, bbox2)
    return ious[0][0].item()


def get_negatives(node: dict, parent: dict, size: tuple, type_: str = '4sibling',
                 all_nodes: list = None, siblings: list = None, childrens: list = None) -> tuple:
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
            count += 1
            if count > max_count:
                return None

        node_new = copy.deepcopy(parent)
        node_new['bounds'] = [s_left, s_top, s_right, s_low]
        return (node, 3, node_new)

    else:  # type_ == '4sibling'
        t = random.randint(0, len(all_nodes) - 1)
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
            t = random.randint(0, len(all_nodes) - 1)
            count += 1
            if count > max_count:
                return None

        return (node, 4, all_nodes[t])


def parse_json_hierarchy(data: list, size: tuple, num: int, all_nodes: list) -> list:
    """
    Parsing json files by including relationships.
    """
    result = []

    def dfs(node: dict, parent: dict = None) -> None:
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
                sample = get_negatives(node_repr, parent_repr, size, '4sibling', all_nodes, siblings, node.get('children'))
                if sample is not None:
                    result.append(sample)

        if 'children' in node:
            for child in node['children']:
                if judge(node['bounds'], child['bounds'], size):
                    continue
                dfs(child, node)

    for entry in data:
        dfs(entry)

    return result


def unif_bbox(bbox: list, size_source: tuple, scale: int) -> list:
    """
    Normalize the bbox.
    """
    w, h = size_source[0], size_source[1]
    return [
        int((bbox[0] / w) * scale),
        int((bbox[1] / h) * scale),
        int((bbox[2] / w) * scale),
        int((bbox[3] / h) * scale)
    ]


def get_relation_prediction_data(data_path: str, save_path: str) -> None:
    train_pairs = []
    Uni_Labels_List = []
    part = 1
    a, b, c, d = 0, 0, 0, 0
    with open(data_path, 'r') as f:
        data = json.load(f)
        for img in tqdm(data):
            sem_json = 'rico_semantic/rico_dataset_v0.1_semantic_annotations/semantic_annotations/' + img.replace('jpg', 'json')
            if not os.path.exists(sem_json):
                continue

            clean_data, all_nodes = get_clean_data(sem_json)
            size = get_resolution('rico_semantic/rico_dataset_v0.1_semantic_annotations/semantic_annotations/' + img.replace('jpg', 'png'))

            output = parse_json_hierarchy(clean_data, size, 2, all_nodes)

            for item in output:
                sample = {}
                sample['image_path'] = 'dataset/Rico/' + img
                # size_real = get_resolution(sample['image_path'])
                scale = 1000
                node1_bbox = unif_bbox(item[0]['bounds'], size, scale)
                node1_label = item[0]['componentLabel']
                node2_bbox = unif_bbox(item[2]['bounds'], size, scale)
                node2_label = item[2]['componentLabel']
                node2_class = item[2]['iconClass']
                relation = item[1]

                if node1_label not in Uni_Labels_List:
                    Uni_Labels_List.append(node1_label)
                if node2_label not in Uni_Labels_List:
                    Uni_Labels_List.append(node2_label)

                if node1_bbox == node2_bbox:
                    continue

                node1_type = node1_label if node1_label != '' else node2_class
                node2_type = node2_label if node2_label != '' else node2_class
                n1 = random.randint(0, len(QUESTION_PROMPT_FOR_RELATION_PREDICTION) - 1)
                n2 = random.randint(0, len(SIBLING_RELATIONSHIP_ANSWER_PROMPT_FOR_RELATION_PREDICTION) - 1)
                n3 = random.randint(0, len(CONTAINMENT_RELATIONSHIP_ANSWER_PROMPT_FOR_RELATION_PREDICTION) - 1)
                n4 = random.randint(0, len(NO_CONTAINMENT_RELATIONSHIP_ANSWER_PROMPT_FOR_RELATION_PREDICTION) - 1)
                n5 = random.randint(0, len(OTHERS) - 1)

                question = QUESTION_PROMPT_FOR_RELATION_PREDICTION[n1].format(node1_bbox, node2_bbox)

                if relation == 0:
                    a += 1
                    parent_bbox = unif_bbox(item[-1]['bounds'], size, scale)
                    parent_type = item[-1]['componentLabel']
                    assert (parent_type is not None) and (parent_type != '')
                    answer = SIBLING_RELATIONSHIP_ANSWER_PROMPT_FOR_RELATION_PREDICTION[n2].format(
                        bbox1=node1_bbox, type1=node1_type, bbox2=node2_bbox, type2=node2_type, bbox3=parent_bbox, type3=parent_type)

                elif relation == 1 or relation == 2:
                    b += 1
                    if relation == 1:
                        node1_bbox, node1_type, node2_bbox, node2_type = node2_bbox, node2_type, node1_bbox, node1_type
                    answer = CONTAINMENT_RELATIONSHIP_ANSWER_PROMPT_FOR_RELATION_PREDICTION[n3].format(
                        bbox1=node1_bbox, type1=node1_type, bbox2=node2_bbox, type2=node2_type)

                elif relation == 3:
                    c += 1
                    answer = NO_CONTAINMENT_RELATIONSHIP_ANSWER_PROMPT_FOR_RELATION_PREDICTION[n4].format(
                        bbox1=node1_bbox, type1=node1_type, bbox2=node2_bbox)

                elif relation == 4:
                    d += 1
                    answer = OTHERS[n5].format(
                        bbox1=node1_bbox, type1=node1_type, bbox2=node2_bbox, type2=node2_type)

                sample['question'] = question
                sample['answer'] = answer

                train_pairs.append(sample)

                if len(train_pairs) == 100000:
                    random.shuffle(train_pairs)
                    print(f'### Saving {(part - 1) * 100000}-{part * 100000} samples.......')
                    with open(save_path + f'{part}_' + 'relation_train.json', 'w') as f:
                        f.write(json.dumps(train_pairs))
                    train_pairs = []
                    part += 1
    count = len(train_pairs) + (part - 1) * 100000

    print(f'label=0:{a / count}, label=1,2:{b / count}, label=3:{c / count}, label=4:{d / count}')
    print(f'full num: {count}')
    print(f'{a}  {b}  {c}  {d}')
    print(Uni_Labels_List)

    random.shuffle(train_pairs)
    with open(save_path + f'{part}_' + 'relation_train.json', 'w') as f:
        f.write(json.dumps(train_pairs))


QUESTION_PROMPT_FOR_RELATION_PREDICTION = [
    "Output the relationship between the elements at {} and {}.(with XML format)",
    "How are the items situated in {} and {} related to one another?(with XML format)",
    "How do the elements in {} and {} relate to each other?(with XML format)",
    "There are two components at {} and {}. What is their relationship?(with XML format)",
    "What is the relationship between the item at {} and {}?(with XML format)"
]
# label=0
SIBLING_RELATIONSHIP_ANSWER_PROMPT_FOR_RELATION_PREDICTION = [
    "Related:\n\
<{type3}>\n\
    <box>{bbox3}</box>\n\
    <{type1}>\n\
        <box>{bbox1}</box>\n\
    </{type1}>\n\
    <{type2}>\n\
        <box>{bbox2}</box>\n\
    </{type2}>\n\
</{type3}>"
]
# label=1, 2
CONTAINMENT_RELATIONSHIP_ANSWER_PROMPT_FOR_RELATION_PREDICTION = [
    "Related:\n\
<{type2}>\n\
    <box>{bbox2}</box>\n\
    <{type1}>\n\
        <box>{bbox1}</box>\n\
    </{type1}>\n\
</{type2}>"
]
# label=3
NO_CONTAINMENT_RELATIONSHIP_ANSWER_PROMPT_FOR_RELATION_PREDICTION = [
    "Unrelated:\n\
<{type1}>\n\
    <box>{bbox1}</box>\n\
</{type1}>,\n\
<Unknown>\n\
    <box>{bbox2}</box>\n\
</Unknown>"
]
# no direct spatial relationship
OTHERS = [
    "Unrelated:\n\
<{type1}>\n\
    <box>{bbox1}</box>\n\
</{type1}>,\n\
<{type2}>\n\
    <box>{bbox2}</box>\n\
</{type2}>"
]


if __name__ == '__main__':
    save_path = 'SpatialRelationPrediction/'
    get_relation_prediction_data('dataset/rico_sem_train_index.json', save_path)