import cv2
import mediapipe as mp

# Инициализация MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Камера
cap = cv2.VideoCapture(0)

def is_thumb_up(landmarks):
    # Координаты точек большого пальца
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    
    # Координаты других пальцев (кончики)
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    # Проверка: большой палец выше (y меньше), а остальные сжаты (y больше)
    thumb_up = thumb_tip.y < thumb_ip.y
    fingers_down = all(landmarks[i].y > landmarks[i - 2].y for i in [8, 12, 16, 20])
    
    return thumb_up and fingers_down

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Переворачиваем для удобства и переводим в RGB
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if is_thumb_up(hand_landmarks.landmark):
                gesture = "👍 Thumbs Up!"

    # Отображение текста
    if gesture:
        cv2.putText(frame, gesture, (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), 4)

    cv2.imshow("Thumbs Up Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
