import pandas as pd
import numpy as np
import mediapipe as mp
import cv2
import csv
import os

data = pd.read_csv('train.csv')

csv_filename = 'hand_landmarks_dataset.csv'

#Инициализация библиотеки mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
                       max_num_hands=1, #Сча тут ода рука на изображении изется, если че можно сделать 2, я пока хз сколько рук на фотках
                       min_detection_confidence=0.5, #Коэффициент детекции - надо будет поиграться, чтобы добиться четкого распознавания рук даже в темноте.
                       static_image_mode=True) #Флаг того, что мы обрабатываем фотки
mp_draw = mp.solutions.drawing_utils

#Создаем список всех признаков
headers = ['label', 'label_id', 'path']
for i in range(21):
    headers += [f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z']

with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)

    for root, dirs, files in os.walk('dataset'):
        dirs.sort()  # Сортируем папки по имени
        files.sort()  # Сортируем файлы в папке
        for file in files:
            image_path = os.path.join(root, file)
            label_id = data.loc[data['video_id'] == int(os.path.basename(root)), 'label_id'].values[0]  # Название подпапки — это метка класса
            label = data.loc[data['video_id'] == int(os.path.basename(root)), 'label'].values[0]
            path = int(os.path.basename(root))

            image = cv2.imread(image_path)
            if image is None:
                continue  # если картинка не загрузилась

            height, width, _ = image.shape
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0].landmark
                row = [label, label_id, path]  # начинаем строку с метки
                for lm in landmarks:
                    row += [lm.x * width, lm.y * height, lm.z * width]
                writer.writerow(row)
            else:
                row = [label, label_id, path]
                row += [0]*63
                writer.writerow(row)