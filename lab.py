import cv2
import mediapipe as mp
from collections import deque
import pyautogui
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
from tensorflow_addons.optimizers import AdamW 
import joblib

class_names = ['Doing other things', 'No gesture',
    'Swiping Down', 'Swiping Left', 'Swiping Right', 'Swiping Up']

def focal_loss_fixed(y_true, y_pred, gamma=2., alpha=0.25):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    cross_entropy = -y_true * tf.math.log(y_pred)
    weight = alpha * tf.math.pow(1 - y_pred, gamma)
    loss = weight * cross_entropy
    return tf.reduce_sum(loss, axis=1)

allowed_points = [0, 1, 5, 6, 8, 9, 10, 12, 13, 17]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

model = load_model("model_good.keras")


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
        data_frame = []
        for hand_landmarks in results.multi_hand_landmarks:
            # for i in [0, 1, 5, 6, 8, 9, 10, 12, 13, 17]:
            #     data_frame.extend([-hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y])
            for landmark in hand_landmarks.landmark:
                data_frame.extend([-landmark.x*176, landmark.y*100, landmark.z*176])
        return np.array(data_frame)
    return None

buffer = deque(maxlen=37)

minmax = joblib.load('minmax.save')

last_gesture_time = 0
debounce_delay = 1.0
prediction_buffer = deque(maxlen=5)
gesture_threshold = 7
confidence_threshold = 0.5
confidence = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    data_frame = process_frame(frame)
    gesture = ""
    class_id = None

    if data_frame is not None:
        buffer.append(data_frame)
        
        if len(buffer) == 37:
            scaledBuffer = minmax.transform(buffer)
            input_data = np.array(scaledBuffer).reshape(1, 37, 42)
            prediction = model.predict(input_data)
            confidence = np.max(prediction)
            if confidence > 0.1:
                class_id = np.argmax(prediction)
                prediction_buffer.append(class_id)
                
            
            # if (time.time() - last_gesture_time) > debounce_delay:
            #     last_gesture_time = time.time()
            print(f"Жест: {class_id}, уверенность: {prediction} ")

            if class_id == 4:
                pyautogui.press('right')
            elif class_id == 3:
                pyautogui.press('left')
            elif class_id == 2:
                pyautogui.scroll(-100)
            elif class_id == 5:
                pyautogui.scroll(100)
                

    if gesture:
        cv2.putText(frame, gesture + ' ' + str(confidence), (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), 4)

    cv2.imshow("Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
