from inference_sdk import InferenceHTTPClient
from IPython.display import Image, display
from roboflow import Roboflow
from shapely.geometry import Point, Polygon
from typing import List, Union
import copy
import json
import numpy as np
import os
import supervision as sv
from PIL import Image
from dataclasses import dataclass

rf = Roboflow(api_key=os.environ['ROBOFLOW_API_KEY'])

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.environ['ROBOFLOW_API_KEY']
)

LABELS_FINGER_TIPS: List[str] = ["1_3", "2_4", "3_4", "4_4", "5_4"]
DISPLAY_LABELS_FINGER: List[str] = ["thumb", "index", "middle", "ring", "pinky"]
FINGER_NAMES_TEXT_SCALE = 0.8
FINGER_DICT = dict(zip(LABELS_FINGER_TIPS, DISPLAY_LABELS_FINGER))
COLOR_FINGER_TIPS = [sv.Color.GREEN, sv.Color.BLUE, sv.Color.RED, sv.Color.YELLOW, sv.Color.ROBOFLOW]
KEY_CENTER_CROSS_COLOR = sv.Color.WHITE
KEY_CENTER_CROSS_LENGTH = 10
KEY_CENTER_CROSS_THICKNESS = 4
KEY_THICKNESS  = 2
ROW_COLORS = [sv.Color.RED, sv.Color.GREEN, sv.Color.BLUE, sv.Color.YELLOW, sv.Color.ROBOFLOW, sv.Color.WHITE]
KEY_EVENT_TEXT_SCALE = 1.0

KEYBOARD_LAYOUT = [
    ["right", "down", "left", "control", "fn", "command", "key.space", "command", "option", "control"],
    ["end", "up", "shift", "/", ".", ",", "m", "n", "b", "v", "c", "x", "z", "shift"],
    ["home", "enter", "'", ";", "l", "k", "j", "h", "g", "f", "d", "s", "a", "caps_lock"],
    ["page_down", "\\", "]", "[", "p", "o", "i", "u", "y", "t", "r", "e", "w", "q", "tab"],
    ["page_up", "backspace", "=", "-", "0", "9", "8", "7", "6", "5", "4", "3", "2", "1", "`"],
    ["del", "f12", "f11", "f10", "f9", "f8", "f7", "f6", "f5", "f4", "f3", "f2", "f1", "esc"]
]

POLYGON_LAYOUT = [[None] * 5 for _ in range(15)]

def find_key_in_layout(key: str) -> tuple[int, int] | None:
    if (key == None):
        return None
    
    for row in range(len(KEYBOARD_LAYOUT)):
        for col in range(len(KEYBOARD_LAYOUT[row])):
            if KEYBOARD_LAYOUT[row][col].lower() == key.lower():
                return row, col
    return None

def print_result(result: sv.Detections):
    print(json.dumps(result,indent=2))

def infer_image(input:  Union[np.ndarray, Image.Image, str], model_id: str) -> dict | list[dict]:
    result = CLIENT.infer(input, model_id)
    
    return result

def infer_video(video_path: str,  project_id: str, model_version: str, fps: int = 24) -> dict | list[dict]:
    project = rf.workspace().project(project_id)
    model = project.version(model_version).model
    job_id, signed_url, expire_time = model.predict_video(
        video_path,
        fps=fps,
        prediction_type="batch-video",
        
    )

    results = model.poll_until_video_results(job_id)

    return results


def add_suffix_to_image_path(image_path: str, suffix: str) -> str:
    parts = image_path.split(".")
    return f"{parts[0]}-{suffix}.{parts[1]}"

def save_image(annotated_image: np.ndarray, target_dir: str, filename: str | None = None):
    if filename is None:
        filename = add_suffix_to_image_path(image_path, "annotated")
    
    with sv.ImageSink(target_dir_path=target_dir, overwrite=False) as sink:
        sink.save_image(image=annotated_image,image_name=filename)
        display(Image(filename=os.path.join(target_dir, filename)))

def filter_keypoints_by_class_name(inference_data: dict, class_names: list[str]):
    """Filter keypoints by class name.

    Args: 
        detection_data: detection data
        class_names: class names to filter

    Returns:
        filtered_detection_data: filtered
    """
    filtered_detection_data = copy.deepcopy(inference_data)
    for prediction in filtered_detection_data['predictions']:
        filtered_keypoints = [kp for kp in prediction['keypoints'] if kp['class_name'] in set(class_names) ]
        prediction['keypoints'] = filtered_keypoints.copy()

    return filtered_detection_data


def keypoint_to_detection(keypoint: dict):
    return sv.Detections(xyxy=np.array([[keypoint["x"], keypoint["y"], keypoint["x"], keypoint["y"]]]))

def annotate_hands(image: np.ndarray, inference: Union[dict, List[dict]]) -> np.ndarray:
    # roboflow_format = {
    #     "predictions": inference['predictions'],
    #     "image": {"width": image.shape[1], "height": image.shape[0]}
    # }
    # detections = sv.Detections.from_inference(roboflow_format)

    key_points = sv.KeyPoints.from_inference(inference)
    vertex_annotator = sv.VertexAnnotator(
        color=sv.Color.BLACK,
        radius=3
    )
    annotated_image = vertex_annotator.annotate(
        scene=image,
        key_points=key_points
    )

    # Add keypoints finger tips
     # Annotate finger names
    
    finger_tips_keypoints = sv.KeyPoints.from_inference(filter_keypoints_by_class_name(inference, LABELS_FINGER_TIPS))
    vertex_label_annotator = sv.VertexLabelAnnotator(
        text_color=sv.Color.BLACK,
        border_radius=10,
        text_padding=0,
        color=COLOR_FINGER_TIPS
    )
    annotated_image = vertex_label_annotator.annotate(
        scene=annotated_image,
        key_points=finger_tips_keypoints,
        labels=["  ", "  ", "  ", "  ", "  "]
    )

    # Add finger name labels
    keypoints_finger_names = ["1_2", "2_1", "3_1", "4_1", "5_1"]
    finger_names_keypoints = sv.KeyPoints.from_inference(filter_keypoints_by_class_name(inference, keypoints_finger_names))
    vertex_label_annotator = sv.VertexLabelAnnotator(
        text_color=sv.Color.BLACK,
        color=COLOR_FINGER_TIPS,
        border_radius=1,
        text_scale=FINGER_NAMES_TEXT_SCALE,
        text_padding=5,
    )
    annotated_image = vertex_label_annotator.annotate(
        scene=annotated_image,
        key_points=finger_names_keypoints,
        labels=["pinky", "ring",  "middle", "index" ,"thumb"]
    )

    return annotated_image

def assign_row(y_value):
    if y_value <= 0.1666666667:
        return 0
    elif y_value <= 0.3333333333:
        return 1
    elif y_value <= 0.5:
        return 2
    elif y_value <= 0.6666666667:
        return 3
    elif y_value <= 0.8333333333:
        return 4
    else:
        return 5
    
@dataclass
class KeyEvent:
    hand: dict | None 
    finger: str | None
    key: str


def annotate_key_events(image: np.ndarray, key_event: KeyEvent) -> np.ndarray:
    rect=sv.Rect(25, 50, 500, 80)
    annotated_image = sv.draw_filled_rectangle(
        scene=image.copy(), 
        rect=rect, 
        color=sv.Color.WHITE,
    )
    annotated_image = sv.draw_rectangle(
        annotated_image,
        rect=rect,
        color=sv.Color.BLACK,
        thickness=2
    )

    if key_event is None:
        # if no key event, return the image with empty rectangle
        return image
    
    text_anchor = sv.Point(260, 85)
    #print(FINGER_DICT)
    if (key_event.finger == "3_4"):
        finger_name = "thumb"
    elif(key_event.finger == "4_4"):
        finger_name = "index"
    elif (key_event.finger == "5_4"):
        finger_name = "middle"
    elif (key_event.finger == "2_4"):
        finger_name = "ring"
    else:
        finger_name = "pinky"

    used_right_finger = None
    
    if (key_event.key in ["1", "q", "a", "z"]):
        if (finger_name == "pinky"):
            used_right_finger = "Nice!"
        else:
            used_right_finger = "Wrong!"
    elif (key_event.key in ["Key.space"]):
        if (finger_name == "thumb"):
            used_right_finger = "Nice!"
        else:
            used_right_finger = "Wrong!"
    if (used_right_finger is None):
        text = f"{finger_name} pressed {key_event.key} key"
    else:
        text = f"{used_right_finger} {finger_name} pressed {key_event.key} key"
   
    annotated_image = sv.draw_text(
        scene=annotated_image, 
        text=text, 
        text_anchor=text_anchor,
        text_color=sv.Color.BLACK, 
        text_scale=KEY_EVENT_TEXT_SCALE, 
        text_thickness=3
    )
    
    return annotated_image

def build_polygon_layout(keys_polygons: np.ndarray) -> List[List[np.array]]:
    """
    Build polygons of keys on the keyboard.
    """
    polygon_centers: list[Point] = [sv.get_polygon_center(polygon=polygon) for polygon in keys_polygons]

    normalized_centers = normalize_points(polygon_centers)
    
    # zip polygon centers with normalized centers
    normalized_polygon_centers = list(zip(polygon_centers, normalized_centers))
    
    rows = [[], [], [], [], [], []]

    polygon_layout = [[None] * 21 for _ in range(6)]
    for center, normalized_center in normalized_polygon_centers:
        row = assign_row(normalized_center[1])
        rows[row].append(center)

    for y, row in enumerate(rows):
        for x, point in enumerate(sort_keys_left_to_right(row)):
            polygon_index = find_polygon_by_point(keys_polygons, point)
            print(x, y)
            polygon_layout[y][x] = keys_polygons[polygon_index]

    return polygon_layout


def annotate_keys(image: np.ndarray, inference_keyboard: Union[dict, List[dict]], keys_polygons: np.ndarray) -> np.ndarray:
    """
    Draw polygons of keys on the keyboard.
    """
    annotated_image = image.copy()


    polygon_centers: list[Point] = [sv.get_polygon_center(polygon=polygon) for polygon in keys_polygons]

    normalized_centers = normalize_points(polygon_centers)
    
    # zip polygon centers with normalized centers
    normalized_polygon_centers = list(zip(polygon_centers, normalized_centers))

    rows = [[], [], [], [], [], []]

    for center, normalized_center in normalized_polygon_centers:
        row = assign_row(normalized_center[1])
        rows[row].append(center)

    for y, row in enumerate(rows):
        for x, point in enumerate(sort_keys_left_to_right(row)):
            polygon_index = find_polygon_by_point(keys_polygons, point)
            if (polygon_index == -1):
                continue
            if (y == 0):
                # # space
                # if (x <= 5):
                #     # "RIGHT, DOWN, LEFT, CTRL, FN, CMD"
                #     # PINKY
                #     annotated_image = sv.draw_polygon(
                #         scene=annotated_image,
                #         polygon=keys_polygons[polygon_index],
                #         color=sv.Color.GREEN,
                #         thickness=3
                #     )   
                if (x == 6):
                    # SPACE
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.ROBOFLOW,
                        thickness=3
                    )
                # elif (x == 7):
                #     # CMD
                #     annotated_image = sv.draw_polygon(
                #         scene=annotated_image,
                #         polygon=keys_polygons[polygon_index],
                #         color=sv.Color.RED,
                #         thickness=3
                #     )
                # else:
                #     # OPTION, CTRL
                #     annotated_image = sv.draw_polygon(
                #         scene=annotated_image,
                #         polygon=keys_polygons[polygon_index],
                #         color=sv.Color.YELLOW,
                #         thickness=3
                #     )

            elif (y == 1):
                if (x <= 4):
                    # "QWERT"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.RED,
                        thickness=3
                    )
                if (x == 5):
                    # "."
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.GREEN,
                        thickness=3
                    )
                elif (x == 6):
                   # ","
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.GREEN,
                        thickness=3
                    )
                elif (x == 7 or x == 8):
                    # "M,N"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.RED,
                        thickness=3
                    )
                elif (x == 9 or x == 10):
                    # "B,V"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.ROBOFLOW,
                        thickness=3
                    )
                elif (x == 11):
                    # "C"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.GREEN,
                        thickness=3
                    )
                elif (x == 12):
                    # "X"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.YELLOW,
                        thickness=3
                    )
                elif (x >= 13):
                    # "Z, Right Shift"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.BLUE,
                        thickness=3
                    )
            elif (y == 2):
                if (x <= 3):
                    # "ASDFG"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.RED,
                        thickness=3
                    )
                elif (x == 4):
                    # "L"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.GREEN,
                        thickness=3
                    )
                elif (x == 5):
                    # "K"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.GREEN,
                        thickness=3
                    )
                elif (x == 6 or x == 7):
                    # "J,H"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.RED,
                        thickness=3
                    )
                elif (x == 8 or x == 9):
                    # "G,F"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.ROBOFLOW,
                        thickness=3
                    )
                elif (x == 10):
                    # "D"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.GREEN,
                        thickness=3,
                    )
                elif (x == 11):
                    # "S"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.YELLOW,
                        thickness=3
                    )
                elif (x >= 12):
                    # "A, CAPS"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.BLUE,
                        thickness=3
                    )
            elif (y == 3):
                if (x <= 4):
                    # "ZXCVB"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.RED,
                        thickness=3
                    )
                elif (x == 5):
                    # "O"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.GREEN,
                        thickness=3
                    )
                elif (x == 6):
                    # "I"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.GREEN,
                        thickness=3
                    )
                elif (x == 7 or x == 8):
                    # "U,Y"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.RED,
                        thickness=3
                    )
                elif (x == 9 or x == 10):
                    # "T,R"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.ROBOFLOW,
                        thickness=3
                    )
                elif (x == 11):
                    # "E"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.GREEN,
                        thickness=3
                    )
                elif (x == 12):
                    # "W"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.YELLOW,
                        thickness=3
                    )
                elif (x >= 13):
                    # "Q, TAB"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.BLUE,
                        thickness=3
                    )
            elif (y == 4):
                if (x <= 4):
                    # "12345"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.RED,
                        thickness=3
                    )
                elif (x == 5):
                    # "9"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.GREEN,
                        thickness=3
                    )
                elif (x == 6):
                    # "8"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.GREEN,
                        thickness=3
                    )
                elif (x == 7 or x == 8):
                    # "7,6"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.RED,
                        thickness=3
                    )
                elif (x == 9 or x == 10):
                    # "5,4"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.ROBOFLOW,
                        thickness=3
                    )
                elif (x == 11):
                    # "3"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.GREEN,
                        thickness=3
                    )
                elif (x == 12):
                    # "2"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.YELLOW,
                        thickness=3
                    )
                elif (x >= 13):
                    # "1, ESC"
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=keys_polygons[polygon_index],
                        color=sv.Color.BLUE,
                        thickness=3
                    )

                
                


            # annotated_image = sv.draw_line(
            #     scene=annotated_image,
            #     start=sv.Point(int(point.x - KEY_CENTER_CROSS_LENGTH), int(point.y -KEY_CENTER_CROSS_LENGTH)),
            #     end=sv.Point(int(point.x + KEY_CENTER_CROSS_LENGTH), int(point.y + KEY_CENTER_CROSS_LENGTH)),
            #     color=ROW_COLORS[i],
            #     thickness=KEY_CENTER_CROSS_THICKNESS
            # )
            
            
    # for point in polygon_centers:
    #     annotated_image = sv.draw_line(
    #         scene=annotated_image, 
    #         start=sv.Point(int(point.x - KEY_CENTER_CROSS_LENGTH), int(point.y + KEY_CENTER_CROSS_LENGTH)),
    #         end=sv.Point(int(point.x + KEY_CENTER_CROSS_LENGTH), int(point.y - KEY_CENTER_CROSS_LENGTH)),
    #         color=KEY_CENTER_CROSS_COLOR, 
    #         thickness=2
    #     )


    
    # find the key at the bottom left corner
    # leftmost_key_index = np.argmin([point.x for point in polygon_centers])


    # leftmost_polygon = keys_polygons[leftmost_key_index]
    # annotated_image = sv.draw_polygon(
    #     scene=annotated_image,
    #     polygon=leftmost_polygon,
    #     color=sv.Color.YELLOW,
    #     thickness=2
    # )

    return annotated_image

def sort_keys_left_to_right(polygon_centers: list[Point]) -> list[Point]:
    return sorted(polygon_centers, key=lambda point: point.x)

def annotate_keys_under_fingertips(image: np.ndarray, inference_keyboard: Union[dict, List[dict]], inference_hands: Union[dict, List[dict]], keys_polygons: np.ndarray) -> np.ndarray:
    """
    Draw polygons of keys on the keyboard.
    """    
    annotated_image = image.copy()
    keypoints_finger_tips = sv.KeyPoints.from_inference(filter_keypoints_by_class_name(inference_hands, LABELS_FINGER_TIPS))

    for key_polygon in keys_polygons:
        polygon_zone = sv.PolygonZone(
            polygon=key_polygon,
        )
        for hand in keypoints_finger_tips:
            i = 0
            for  finger_tip in hand[0]:
                #print(finger_tip)  
                detections = sv.Detections(
                    xyxy=np.array([[int(finger_tip[0]), int(finger_tip[1]), int(finger_tip[0]), int(finger_tip[1])]]),
                    data={"class_name": np.array([LABELS_FINGER_TIPS[i]])}
                )
                
                is_in_zone = polygon_zone.trigger(detections)
                if (is_in_zone):
                    #print(f"finger {detections.data['class_name'][0]} is in zone")
                    annotated_image = sv.draw_polygon(
                        scene=annotated_image,
                        polygon=key_polygon,
                        color=sv.Color.RED,
                        thickness=2
                    )
                i += 1
    
    return annotated_image

def find_fingertip_in_polygon(key_polygon: np.array, inference_hands: Union[dict, List[dict]]) -> np.ndarray:
    """
    Draw polygons of keys on the keyboard.
    """
    
    keypoints_finger_tips = sv.KeyPoints.from_inference(filter_keypoints_by_class_name(inference_hands, LABELS_FINGER_TIPS))

    polygon_zone = sv.PolygonZone(
        polygon=key_polygon,
    )
    for hand in keypoints_finger_tips:
        #print(hand)
        i = 0
        for  finger_tip in hand[0]:
            # if (i == 2):
            #     correct_index = 4
            # elif (i == 4):
            #     correct_index = 2
            # else:
            #     correct_index = i
            
            detections = sv.Detections(
                xyxy=np.array([[int(finger_tip[0]), int(finger_tip[1]), int(finger_tip[0]), int(finger_tip[1])]]),
                data={"class_name": np.array([LABELS_FINGER_TIPS[i]])}
            )
            
            is_in_zone = polygon_zone.trigger(detections)
            if (is_in_zone):
                #print(f"finger {detections.data['class_name'][0]} is in zone")
                return hand, LABELS_FINGER_TIPS[i]
            i += 1
    
    return None

def find_finger_tip_in_polygon(polygon_zone: sv.PolygonZone, inference_keyboard: Union[dict, List[dict]], inference_hands: Union[dict, List[dict]]) -> np.ndarray:
    """
    Draw polygons of keys on the keyboard.
    """

    #polygon_zones: List[sv.PolygonZone] = []
    
    keypoints_finger_tips = sv.KeyPoints.from_inference(filter_keypoints_by_class_name(inference_hands, LABELS_FINGER_TIPS))

    for hand in keypoints_finger_tips:
        i = 0
        for finger_tip in hand[0]:
            #print(finger_tip)  
            finger_tip_point = Point(finger_tip[0], finger_tip[1])
            if (is_point_in_polygon(polygon_zone, finger_tip_point)):
                return hand, finger_tip
    
    return None

def build_keys_polygons(inference: Union[dict, List[dict]]) -> np.ndarray[np.array]:
    polygons: np.ndarray[np.array] = []
    for prediction in inference['predictions']:
        points = prediction['points']
        polygon = np.array([[int(point["x"]), int(point["y"])] for point in points])
        polygons.append(polygon)
    
    return polygons

def find_polygon_by_point(polygons: List[np.ndarray], point: sv.Point) -> int:
    for i, polygon in enumerate(polygons):
        if Point(point.x, point.y).within(Polygon(polygon)):
            return i
    return -1

def is_point_in_polygon(polygon: np.ndarray, point: sv.Point) -> int:
    if Point(point.x, point.y).within(Polygon(polygon)):
        return True
    return False

# Function to sort points in the rectangular order



  
def normalize_points(points: list[Point]):
    
    # Find the min and max for x and y
    min_x = min(point.x for point in points)
    max_x = max(point.x for point in points)
    min_y = min(point.y for point in points)
    max_y = max(point.y for point in points)

    # # Calculate the ranges
    range_x = max_x - min_x
    range_y = max_y - min_y

    # # Normalize points
    normalized_points = [[((point.x - min_x) / range_x), ( (point.y - min_y) / range_y)] for point in points]

    return normalized_points

