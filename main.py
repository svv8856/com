import pyautogui
import cv2
import numpy as np
import time

# Путь к изображению с крыльями
image_to_find = 'wings_image.png'

# Настройка размера экрана для захвата
screen_width, screen_height = pyautogui.size()

# Функция для выполнения поиска и клика
def find_and_click():
    screen = np.array(pyautogui.screenshot())  # Получаем скриншот экрана
    gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(image_to_find, 0)  # Загружаем изображение для поиска (в сером)

    # Инициализация SIFT детектора
    sift = cv2.SIFT_create()
    
    # Находим ключевые точки и дескрипторы
    keypoints_screen, descriptors_screen = sift.detectAndCompute(gray_screen, None)
    keypoints_template, descriptors_template = sift.detectAndCompute(template, None)

    # Инициализация сопоставления ключевых точек
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Сопоставляем дескрипторы
    matches = bf.match(descriptors_template, descriptors_screen)

    # Сортируем по расстоянию (чем меньше значение, тем лучше совпадение)
    matches = sorted(matches, key=lambda x: x.distance)

    # Проверка на наличие совпадений
    if len(matches) > 10:  # Убедитесь, что количество совпадений достаточное
        # Получаем точку совпадения
        match = matches[0]
        pt = keypoints_screen[match.trainIdx].pt

        # Выполняем клик в найденной точке
        pyautogui.click(pt[0], pt[1])
        print(f"Клик в точке: {pt}")
    else:
        print("Не найдено достаточно совпадений.")

# Главный цикл, который будет постоянно проверять экран и кликать
while True:
    find_and_click()  # Пытаемся найти и кликнуть
    time.sleep(0.05)  # Ждем немного перед следующим поиском (0.1 секунды)
