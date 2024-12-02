import json
import os
import random
from tqdm import tqdm

ICON_TYPE = {
    'ICON_SEND': ['Send Icon', 'Used to send messages or data'],
    'ICON_CALL': ['Call Icon', 'Initiates a phone call'],
    'ICON_STAR': ['Star Icon', 'Marks an item as a favorite'],
    'ICON_PAUSE': ['Pause Icon', 'Pauses a current activity'],
    'ICON_THREE_DOTS': ['Three Dots Icon', 'Indicates more options'],
    'ICON_CALENDAR': ['Calendar Icon', 'Displays calendar events'],
    'ICON_MIC': ['Microphone Icon', 'Used for audio input'],
    'ICON_PLUS': ['Plus Icon', 'Adds a new item or feature'],
    'ICON_PEOPLE': ['People Icon', 'Represents users or contacts'],
    'ICON_SAD_FACE': ['Sad Face Icon', 'Represents a sad face symbol'],
    'ICON_LAUNCH_APPS': ['Launch Apps Icon', 'Opens an app drawer'],
    'ICON_FACEBOOK': ['Facebook Icon', 'Links to Facebook platform'],
    'ICON_VOLUME_STATE': ['Volume State Icon', 'Indicates audio volume status'],
    'ICON_HEART': ['Heart Icon', 'Represents health, love or favorites'],
    'ICON_TIME': ['Time Icon', 'Displays time-related information'],
    'ICON_CLOUD': ['Cloud Icon', 'Indicates weather, cloud storage or services'],
    'ICON_ARROW_BACKWARD': ['Backward Arrow Icon', 'Navigates to the previous screen'],
    'ICON_SHARE': ['Share Icon', 'Sharing content with others'],
    'ICON_GOOGLE': ['Google Icon', 'Links to Google services'],
    'ICON_SETTINGS': ['Settings Icon', 'Opens configuration options'],
    'ICON_NAV_BAR_CIRCLE': ['Nav Bar Circle Icon', 'Indicates navigation button'],
    'ICON_V_UPWARD': ['Upward Icon', 'Indicates upward movement or scroll'],
    'ICON_MIC_MUTE': ['Muted Microphone Icon', 'Indicates audio input is muted'],
    'ICON_V_FORWARD': ['Forward Icon', 'Indicates forward movement or action'],
    'ICON_PAPERCLIP': ['Paperclip Icon', 'Used to attach files'],
    'ICON_ENVELOPE': ['Envelope Icon', 'Indicates email or message'],
    'ICON_GALLERY': ['Gallery Icon', 'Opens photo or image collection'],
    'ICON_VIDEOCAM': ['Video Camera Icon', 'Used for video capture'],
    'ICON_CHAT': ['Chat Icon', 'Initiates a chat conversation'],
    'ICON_ASSISTANT': ['Assistant Icon', 'Represents a virtual assistant'],
    'ICON_LIST': ['List Icon', 'Displays a list view'],
    'ICON_HOME': ['Home Icon', 'Navigates to the home screen'],
    'ICON_MAGNIFYING_GLASS': ['Magnifying Glass Icon', 'Indicates search functionality'],
    'ICON_NOTIFICATIONS': ['Notifications Icon', 'Displays alerts and notifications'],
    'ICON_PLAY': ['Play Icon', 'Starts media playback'],
    'ICON_LOCATION': ['Location Icon', 'Indicates geographical location'],
    'ICON_STOP': ['Stop Icon', 'Stops current activity'],
    'ICON_INFO': ['Info Icon', 'Displays information'],
    'ICON_V_BACKWARD': ['Backward Icon', 'Indicates backward movement or action'],
    'ICON_NAV_BAR_RECT': ['Nav Bar Rect Icon', 'Represents navigation button'],
    'ICON_X': ['Close Icon', 'Closes a window or dialog'],
    'ICON_CHECK': ['Checkmark Icon', 'Indicates selection completion'],
    'ICON_V_DOWNWARD': ['Downward Icon', 'Indicates downward movement or scroll'],
    'ICON_SUN': ['Sun Icon', 'Indicates screen mode or brightness'],
    'ICON_PERSON': ['Person Icon', 'Represents a user or profile'],
    'ICON_THREE_BARS': ['Three Bars Icon', 'Represents a menu or navigation'],
    'ICON_TAKE_PHOTO': ['Take Photo Icon', 'Initiates taking a photo'],
    'ICON_ARROW_FORWARD': ['Forward Arrow Icon', 'Navigates to the next screen'],
    'ICON_REFRESH': ['Refresh Icon', 'Reloads the current page or content'],
    'ICON_DOWNLOAD': ['A downward arrow', 'Initiates a download process'],
    'ICON_HISTORY': ['A clock symbol representing past records', "To view history or past actions"],
    'ICON_ARROW_DOWNWARD':['Arrow Downward Icon', 'Indicates downward movement or scroll with an arrow symbol'],
    'ICON_QUESTION': ['Question mark icon', 'Symbolizes a question or inquiry about information'],
    'ICON_COMPASS': ["Find direction or navigation", "Indicates navigation or locating purpose"],
    'ICON_EXPAND': ["Expand Icon", "Used to expand a menu or section"],
    'ICON_ARROW_UPWARD': ["An upward-pointing arrow symbol", "Used to indicate upward movement or action"],
    'ICON_SHOPPING_CART': ["A cart filled with goods", "Used to represent shopping or purchases"],
    'ICON_HOME': ["Home Icon", "Back to the main screen"],
    'ICON_THUMBS_UP': ["Thumbs Up Icon", "Indicates approval or like"],
    'ICON_CAST': ["Cast Icon", "Used to cast content to other devices"],
    'ICON_END_CALL': ["End Call Icon", "Ends an ongoing phone call"],
    'ICON_UPLOAD': ["Upload Icon", "Initiates file upload"],
    'ICON_HAPPY_FACE': ["Happy Face Icon", "Represents a smiley face symbol"]
}


def process_bboxdef process_bbox(bbox: List[float], width: int, height: int) -> List[int]:
    """
    Convert bounding box coordinates from normalized to absolute pixel values.

    Args:
        bbox (List[float]): Bounding box in the format [y, x, h, w].
        width (int): Width of the image.
        height (int): Height of the image.

    Returns:
        List[int]: Bounding box in the format [x1, y1, x2, y2].
    """
    new_bbox = [
        int(bbox[1] * width),
        int(bbox[0] * height),
        int((bbox[1] + bbox[3]) * width),
        int((bbox[0] + bbox[2]) * height)
    ]
    return new_bbox  # [x1, y1, x2, y2]

def cal_size(bbox: List[int]) -> int:
    """
    Calculate the area of the bounding box.

    Args:
        bbox (List[int]): Bounding box in the format [x1, y1, x2, y2].

    Returns:
        int: Area of the bounding box.
    """
    a, b, c, d = bbox
    return (c - a) * (d - b)

def cal_percent(bbox: float, image_size: float) -> float:
    """
    Calculate the percentage of a bbox relative to a screenshot image.

    Args:
        bbox (float): bbox size.
        image_size (float): The total screenshot image size.

    Returns:
        float: Percentage representation.
    """

    return (bbox / image_size) * 100 


def normalize_bbox(box: List[int], width: int, height: int) -> List[int]:
    """
    Normalize the bounding box coordinates for minimal compression.

    Args:
        box (List[int]): Bounding box in the format [x1, y1, x2, y2].
        width (int): Width of the image.
        height (int): Height of the image.

    Returns:
        List[int]: Normalized bounding box.
    """
    return [
        box[0] * 1000 // width,
        box[1] * 1000 // height,
        box[2] * 1000 // width,
        box[3] * 1000 // height
    ]


def get_folds(fold: str = '', save_path: str = '', max_rate: float = 0.3) -> None:
    """
    Process JSON metadata files to extract and save training items based on icon types and bounding box constraints.

    Args:
        fold (str): Directory containing the data folds.
        save_path (str): Directory where the processed JSON files will be saved.
        max_rate (float): Maximum allowed percentage area for the bounding box.
    """
    types = []
    part = 0
    train_items = []

    dirs = os.listdir(fold) 

    for dir_ in tqdm(dirs):
        # if dir_ in ['google_apps3', 'google_apps4', 'google_apps1', 'google_apps2']: continue 
        files = os.listdir(fold+dir_) 

        for file in files:
            path = fold + dir_ + '/' + file + '/' + 'meta.json'

            with open(path, 'r') as f:
                data = json.load(f)

                for row in data:
                    item = {}
                    item['image_path'] = row['image_path']
                    bbox_list = row['image/ui_annotations_positions']
                    item['orc_list'] = row['image/ui_annotations_text']
                    item['type_list'] = row['image/ui_annotations_ui_types']
                    map_ = {}

                    for i, x in enumerate(item['type_list']):

                        if x in map_:
                            map_[x].append(bbox_list[i])
                        else:
                            map_[x] = [bbox_list[i]]

                    for x in map_:
                        if len(map_[x]) == 1 and (x not in ['TEXT']):
                            sample = {}
                            sample['image_path'] = item['image_path'] 
                            w, h =  row['image/width'], row['image/height']
                            sample['resolution'] = [w, h]
                            sample['gt'] = process_bbox(map_[x][0], w, h)
                            sample['type'] = x

                            source_bbox = cal_size(sample['gt'])
                            rate = cal_percent(source_bbox, w * h)
                            sample['rate'] = rate
                            sample['gt'] = normalize_bbox(sample['gt'], w, h)

                            if rate <= 0. or rate >= max_rate: continue
                            else:

                                if x not in ICON_TYPE: continue
                                else:
                                    question = f'Give the bounding box reffering to "{ICON_TYPE[x][0].lower()}".'
                                    sample['question'] = question
                                    train_items.append(sample)

                        if len(train_items) == 100000:
                            random.shuffle(train_items)

                            with open(save_path + f'AITW_{part}_small_icon_train.json', 'w') as f:
                                f.write(json.dumps(train_items))
                            part += 1
                            train_items = []

    random.shuffle(train_items)

    with open(save_path + f'AITW_{part}_small_icon_train.json', 'w') as f:
        f.write(json.dumps(train_items))

    print(f'full nums:{len(train_items)+ (part-1)*100000}')

if __name__ == '__main__':

    get_folds(fold='AITW_data_path/', 
                save_path='saving_path/', max_rate=0.3)

