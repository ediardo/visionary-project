from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from pynput import keyboard
import cv2
import supervision as sv
from annotate import *
from functools import partial

import cv2
import time

import cv2.typing

def capture_photo() -> cv2.typing.MatLike:
    """
    Capture a photo using the webcam and return the frame.
    """
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Allow the webcam to warm up
    time.sleep(2)

    # Capture multiple frames to allow the camera to adjust its focus
    for _ in range(30):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            return
    
    # Capture the final frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        return
    
    cap.release()

    return frame

last_pressed_key = None
def on_press(key):
    global last_pressed_key
    try:
        last_pressed_key =  key.char
    except AttributeError:
        last_pressed_key = str(key)

listener = keyboard.Listener(on_press=on_press)
listener.start()
listener.stop


def custom_prediction(predictions_hands:any, video_frame: VideoFrame):
    global predictions_keyboard
    global keys_polygons
    global polygon_layout

    annotated_frame = annotate_keys(
        image=video_frame.image.copy(), 
        inference_keyboard=predictions_keyboard, 
        keys_polygons=keys_polygons
    )

    annotated_frame = annotate_hands(image=annotated_frame, inference=predictions_hands)
    key_event: KeyEvent = None

    key_position_in_layout = find_key_in_layout(key=last_pressed_key)

    if (key_position_in_layout is not None):
        row, col = key_position_in_layout
        pressed_polygon = polygon_layout[row][col]
        matched_finger = find_fingertip_in_polygon(key_polygon=pressed_polygon, inference_hands=predictions_hands)
        
        if (matched_finger is not None):
            hand, finger = matched_finger
            key_event = KeyEvent(hand, finger, last_pressed_key)
        else:
            print("No fingertip found in the key polygon.")

    annotated_frame = annotate_key_events(image=annotated_frame, key_event=key_event)
    cv2.imshow("Predictions", annotated_frame)
    cv2.waitKey(1)

keyboard_image = capture_photo()

predictions_keyboard = infer_image(keyboard_image, "keyboard-segmentation/1")
keys_polygons = build_keys_polygons(inference=predictions_keyboard)
polygon_layout = build_polygon_layout(keys_polygons=keys_polygons)

# initialize a pipeline object
pipeline = InferencePipeline.init(
    api_key=os.environ['ROBOFLOW_API_KEY'],
    model_id="hands-new/11", 
    video_reference=0, 
    on_prediction=custom_prediction,
    active_learning_enabled=False,
    max_fps=14
)
pipeline.start()
pipeline.join()
