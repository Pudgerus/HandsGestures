import cv2
import mediapipe as mp
from collections import deque

# Инициализация MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Камера
cap = cv2.VideoCapture(0)

# 💡 Храним историю X координат запястья для свайпа
positions = deque(maxlen=10)

def is_thumb_up(landmarks):
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    
    # Остальные пальцы (кончики)
    fingers_down = all(landmarks[i].y > landmarks[i - 2].y for i in [8, 12, 16, 20])
    
    return thumb_tip.y < thumb_ip.y and fingers_down

def is_middle_up(landmarks):
    middle_tip = landmarks[12]
    middle_ip = landmarks[10]

    fingers_down = all(landmarks[i].y > landmarks[i - 2].y for i in [8, 16, 20])
    # thumb_down = landmarks[4].x > landmarks[2].x

    return middle_tip.y < middle_ip.y and fingers_down

def swipe(landmarks):
    # Добавляем координату X запястья
    positions.append(landmarks[0].x)

    if len(positions) == positions.maxlen:
        delta = positions[-1] - positions[0]
        if delta > 0.2:
            return "Swipe Right"
        elif delta < -0.2:
            return "Swipe Left"
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if is_thumb_up(hand_landmarks.landmark):
                gesture = "👍 Thumbs Up!"
            elif is_middle_up(hand_landmarks.landmark):
                gesture = "Fuck you"
            else:
                swipe_gesture = swipe(hand_landmarks.landmark)
                if swipe_gesture:
                    gesture = swipe_gesture

    if gesture:
        cv2.putText(frame, gesture, (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), 4)

    cv2.imshow("Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
