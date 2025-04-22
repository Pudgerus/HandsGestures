import cv2
import mediapipe as mp

#Значится инициализируем MediaPipe
mp_hands = mp.solution.hands
hands = mp_hands.Hands(
    ststic_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def what_gesture(landmarks):
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]

    middle_tip = landmarks[12]
    middle_ip = landmarks[10]

    index_tip = landmarks[8]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    thumb_up = thumb_tip.y < thumb_ip.y
    middle_up = middle_tip.y < middle_ip.y

    fingers_down = all(landmarks[i].y > landmarks[i - 2].y for i in [8, 16, 20])
    


