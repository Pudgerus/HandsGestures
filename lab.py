import cv2
import mediapipe as mp
from collections import deque
import pyautogui
from tensorflow.keras.models import load_model
import numpy as np

class_names = [
    'Doing other things',
    'Drumming Fingers',
    'No gesture',
    'Pulling Hand In',
    'Pulling Two Fingers In',
    'Pushing Hand Away',
    'Pushing Two Fingers Away',
    'Rolling Hand Backward',
    'Rolling Hand Forward',
    'Shaking Hand',
    'Sliding Two Fingers Down',
    'Sliding Two Fingers Left',
    'Sliding Two Fingers Right',
    'Sliding Two Fingers Up',
    'Stop Sign',
    'Swiping Down',
    'Swiping Left',
    'Swiping Right',
    'Swiping Up',
    'Thumb Down',
    'Thumb Up',
    'Turning Hand Clockwise',
    'Turning Hand Counterclockwise',
    'Zooming In With Full Hand',
    'Zooming In With Two Fingers',
    'Zooming Out With Full Hand',
    'Zooming Out With Two Fingers'
]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

model = load_model("best_model.h5")

positions = deque(maxlen=10)
finger_positions = deque(maxlen=10)
middle_index = deque(maxlen=10)

def is_thumb_up(landmarks):
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    
    fingers_down = all(landmarks[i].y > landmarks[i - 2].y for i in [8, 12, 16, 20])
    
    return thumb_tip.y < thumb_ip.y and fingers_down

def is_middle_up(landmarks):
    middle_tip = landmarks[12]
    middle_ip = landmarks[10]

    fingers_down = all(landmarks[i].y > landmarks[i - 2].y for i in [8, 16, 20])
    # thumb_down = landmarks[4].x > landmarks[2].x

    return middle_tip.y < middle_ip.y and fingers_down

def swipe(landmarks):
    positions.append(landmarks[0].x)
    finger_positions.append([landmarks[8].x, landmarks[12].x, landmarks[16].x])

    if len(finger_positions) == finger_positions.maxlen:
        delta = ((finger_positions[-1][0] - finger_positions[0][0]) + (finger_positions[-1][1] - finger_positions[0][1]) +
                 (finger_positions[-1][2] - finger_positions[0][2])) / 3
        if delta > 0.5:
            return "Swipe Right"
        elif delta < -0.5:
            return "Swipe Left"

    if len(positions) == positions.maxlen:
        delta = positions[-1] - positions[0]
        if delta > 0.5:
            return "Swipe Right"
        elif delta < -0.5:
            return "Swipe Left"
        

def scroll(landmarks):
    index_tip = landmarks[8]                                            
    index_ip = landmarks[6]
    middle_tip = landmarks[12]
    middle_ip = landmarks[10]

    two_fingers = index_tip.y < index_ip.y and middle_tip.y < middle_ip.y
    fingers_down = all(landmarks[i].y > landmarks[i - 2].y for i in [16, 20])
    two_fingers_x = index_tip.x > index_ip.x and middle_tip.x > middle_ip.x

    middle_index.append([index_tip.y, middle_tip.y])

    if (two_fingers and fingers_down and len(middle_index) == middle_index.maxlen) or (two_fingers_x and fingers_down and len(middle_index) == middle_index.maxlen):
        delta = ((middle_index[-1][0] - middle_index[0][0]) + (middle_index[-1][1] - middle_index[0][1]))/2
        if delta > 0.2:
            return "Scroll Up"
        elif delta < -0.5:
            return "Scroll Down"

def spacebar(landmarks):
    fingers_down = all(landmarks[i].y > landmarks[i - 3].y for i in [8, 12, 16, 20])

    return fingers_down


def process_frame(image):
    results = hands.process(image)
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks
        data_frame = []
        for landmark in landmarks.landmark:
            data_frame.extend([landmark.x, landmark.y, landmark.z])
        return np.array(data_frame)
    return None

buffer = deque(maxlen=37)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    data_frame = process_frame(frame)

    gesture = ""

    if data_frame is not None:
        buffer.append(data_frame)
        
        if len(buffer) == 37:  # Когда буфер заполнен
            input_data = np.array(buffer).reshape(1, 37, 63)  # (1, 37, 63)
            prediction = model.predict(input_data)
            class_id = np.argmax(prediction)
            gesture = class_names[class_id]
            print(f"Жест: {gesture}")


    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # scrolll = scroll(hand_landmarks.landmark)

            # if is_middle_up(hand_landmarks.landmark):
            #     gesture = "Fuck you"
            # elif scrolll:
            #     gesture = scrolll
            #     if scrolll == "Scroll Down":
            #         pyautogui.scroll(-100)
            #     else:
            #         pyautogui.scroll(100)
            # # elif spacebar(hand_landmarks.landmark):
            # #     gesture = "Stop"
            # #     pyautogui.press("space")
            # else:
            #     swipe_gesture = swipe(hand_landmarks.landmark)
            #     if swipe_gesture:
            #         gesture = swipe_gesture
            #         if swipe_gesture == "Swipe Left":
            #             pyautogui.press('left')
            #         if swipe_gesture == "Swipe Right": 
            #             pyautogui.press('right')

    if gesture:
        cv2.putText(frame, gesture, (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), 4)

    cv2.imshow("Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
