# Импорт необходимых библиотек
import cv2  # Библиотека для обработки изображений и компьютерного зрения
import numpy as np  # Библиотека для работы с массивами и математическими операциями
import math  # Библиотека для математических функций
from pyiArduinoI2Cmotor import *  # Библиотека для работы с двигателями
import time  # Библиотека для работы со временем
import RPi.GPIO as GPIO  # Библиотека для управления GPIO пинами Raspberry Pi

# Определение класса CylinderDetector для обнаружения цилиндров
class CylinderDetector:
    # Метод инициализации класса
    def __init__(self, min_radius=10, max_radius=400, min_area=500, max_area=50000, vertical_threshold=2.0):
        # Инициализация параметров для обнаружения цилиндров
        self.min_radius = min_radius # Минимальный радиус цилиндра
        self.max_radius = max_radius # Максимальный радиус цилиндра
        self.min_area = min_area # Минимальная площадь контура цилиндра
        self.max_area = max_area # Максимальная площадь контура цилиндра
        self.vertical_threshold = vertical_threshold # Порог для определения вертикальности цилиндра

        # Создание структурного элемента для морфологических операций
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Определение цветов для отрисовки (BGR формат)
        self.green = (0, 255, 0)
        self.red = (0, 0, 255)
        self.blue = (255, 0, 0)

        # Определение диапазонов цветов в HSV для маскирования
        self.color_ranges = {
            'red': ([0, 50, 50], [10, 255, 255]),
            'green': ([35, 50, 50], [85, 255, 255]),
            'blue': ([100, 50, 50], [140, 255, 255]),
            'white': ([0, 0, 205], [180, 30, 255])
        }
        # Определение цветов для отрисовки на изображении (BGR формат)
        self.color_draw = {'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0), 'white': (255, 255, 255)}


    def preprocess_image(self, image, color):
        try:
            # Преобразование изображения из BGR в HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Определение нижнего и верхнего порогов для заданного цвета
            lower_color = np.array(self.color_ranges[color][0])
            upper_color = np.array(self.color_ranges[color][1])

            # Создание маски для заданного цвета
            mask = cv2.inRange(hsv, lower_color, upper_color)

            # Дополнительная обработка для красного цвета (так как он охватывает начало и конец цветового круга)
            if color == 'red':
                lower_color2 = np.array([170, 50, 50])
                upper_color2 = np.array([180, 255, 255])
                mask2 = cv2.inRange(hsv, lower_color2, upper_color2)
                mask = cv2.bitwise_or(mask, mask2)

                lower_color3 = np.array([150, 50, 50])
                upper_color3 = np.array([170, 255, 255])
                mask3 = cv2.inRange(hsv, lower_color3, upper_color3)
                mask = cv2.bitwise_or(mask, mask3)

            # Дополнительная обработка для зеленого цвета (для более точного выделения)
            elif color == 'green':
                lower_green2 = np.array([80, 50, 50])
                upper_green2 = np.array([100, 255, 255])
                mask_green2 = cv2.inRange(hsv, lower_green2, upper_green2)
                mask = cv2.bitwise_or(mask, mask_green2)

                lower_green3 = np.array([25, 50, 50])
                upper_green3 = np.array([35, 255, 255])
                mask_green3 = cv2.inRange(hsv, lower_green3, upper_green3)
                mask = cv2.bitwise_or(mask, mask_green3)

                lower_green4 = np.array([85, 50, 50])
                upper_green4 = np.array([100, 255, 255])
                mask_green4 = cv2.inRange(hsv, lower_green4, upper_green4)
                mask = cv2.bitwise_or(mask, mask_green4)

                lower_emerald = np.array([45, 80, 80])
                upper_emerald = np.array([65, 255, 255])
                mask_emerald = cv2.inRange(hsv, lower_emerald, upper_emerald)
                mask = cv2.bitwise_or(mask, mask_emerald)

                lower_emerald2 = np.array([40, 60, 50])
                upper_emerald2 = np.array([70, 255, 200])
                mask_emerald2 = cv2.inRange(hsv, lower_emerald2, upper_emerald2)
                mask = cv2.bitwise_or(mask, mask_emerald2)

                lower_dark_emerald = np.array([40, 50, 0])
                upper_dark_emerald = np.array([70, 255, 100])
                mask_dark_emerald = cv2.inRange(hsv, lower_dark_emerald, upper_dark_emerald)
                mask = cv2.bitwise_or(mask, mask_dark_emerald)

            # Дополнительная обработка для синего цвета
            elif color == 'blue':
                lower_blue2 = np.array([140, 50, 50])
                upper_blue2 = np.array([160, 255, 255])
                mask_blue2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
                mask = cv2.bitwise_or(mask, mask_blue2)

            # Применение морфологических операций для удаления шума и сглаживания маски
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel) # Открытие (удаление мелких объектов)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel) # Закрытие (заполнение мелких дыр)

            return mask, hsv # Возвращаем обработанную маску и изображение в HSV
        except Exception as e:
            print(f"Error during image preprocessing for color {color}: {e}") # Обработка исключений
            return None, None

    # Метод для обнаружения окружностей с помощью преобразования Хафа
    def detect_circles_hough(self, mask):
        try:
            # Применение преобразования 
            circles = cv2.HoughCircles(
                mask, # Входная маска
                cv2.HOUGH_GRADIENT, # Метод обнаружения
                dp=10, # Соотношение между размером изображения и размером окна аккумулятора
                minDist=100, # Минимальное расстояние между центрами обнаруженных окружностей
                param1=100, # верхний порог для детектора краев Canny
                param2=100, # Параметр, уменьшающий радиус обнаружения для более надежных кругов
                minRadius=self.min_radius, # Минимальный радиус окружности
                maxRadius=self.max_radius # Максимальный радиус окружности
            )
            return circles # Возвращаем список обнаруженных окружностей
        except Exception as e:
            print(f"Error during Hough Circle Transform detection: {e}") # Обработка исключений
            return None

    # Метод для обнаружения контуров и фильтрации их как цилиндры
    def detect_contours(self, mask, color):
        # Поиск внешних контуров на маске
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cylinders = [] # Список для хранения обнаруженных цилиндров
        for contour in contours:
            area = cv2.contourArea(contour) # Вычисление площади контура
            # Фильтрация по площади контура
            if self.min_area < area < self.max_area:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Получение ограничивающего прямоугольника для контура
                x, y, w, h = cv2.boundingRect(contour)

                aspect_ratio = float(w) / h # Вычисление соотношения сторон
                # Фильтрация по соотношению сторон
                if 0.3 < aspect_ratio < 2.0:
                    # Фильтрация по высоте и вертикальности
                    if h > w and h / w > self.vertical_threshold:
                        # Нахождение минимального описывающего круга
                        (cx, cy), radius = cv2.minEnclosingCircle(contour)

                        circle_area = math.pi * radius * radius # Вычисление площади описывающего круга
                        fill_ratio = area / circle_area if circle_area > 0 else 0
                        if 0.3 < fill_ratio < 0.9:
                            cylinders.append({ # Добавление информации об обнаруженном цилиндре
                                'contour': contour,
                                'center': (int(cx), int(cy)),
                                'radius': int(radius),
                                'area': area,
                                'bbox': (x, y, w, h),
                                'color': color
                            })

        return cylinders # Возвращаем список обнаруженных цилиндров

    # Метод для отрисовки обнаруженных цилиндров и окружностей на изображении
    def draw_detections(self, image, cylinders, circles=None, center_x=None):
        result = image.copy() 

        # Отрисовка обнаруженных цилиндров
        for cyl in cylinders:
            color = cyl['color']
            draw_color = self.color_draw[color] 

            cv2.drawContours(result, [cyl['contour']], -1, draw_color, 2) 
            cv2.circle(result, cyl['center'], cyl['radius'], draw_color, 2)
            cv2.circle(result, cyl['center'], 3, self.red, -1) 

            x, y, w, h = cyl['bbox']
            cv2.rectangle(result, (x, y), (x + w, y + h), draw_color, 2)

            # Отрисовка расстояния до центра изображения, если оно доступно
            if center_x is not None:
                distance = cyl['center'][0] - center_x
                distance_text = f"Distance: {distance} px"
                cv2.putText(result, distance_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 1)

        # Отрисовка обнаруженных окружностей 
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(result, (x, y), r, self.red, 2)
                cv2.circle(result, (x, y), 2, self.red, 3) 

        return result # Возвращаем изображение с отрисовками

    # Метод для обработки одного кадра
    def process_frame(self, frame):
        all_cylinders = [] # Список для всех обнаруженных цилиндров
        masks = {} # Словарь для хранения масок по цветам

        height, width = frame.shape[:2] # Получение размеров кадра
        center_x = width // 2 # Вычисление центральной координаты X

        # Перебор всех отслеживаемых цветов
        for color in self.color_ranges.keys():
            mask, hsv = self.preprocess_image(frame, color) 

            if mask is None or hsv is None: # Проверка на ошибки при обработке
                continue

            cylinders = self.detect_contours(mask, color) 
            all_cylinders.extend(cylinders) 

            masks[color] = mask 

        return all_cylinders, masks, center_x 

    # Метод для установки угла сервопривода
    def set_angle(self, angle):
        duty = (angle / 18) + 2
        pwm.ChangeDutyCycle(duty) 
        time.sleep(0.5) 

# Основная функция программы
def main():
    # Инициализация переменных с статусом цилиндров
    cb = 2
    ck = 2
    cg = 2
    cw = 2
    n = 0
    # Параметры детектора цилиндров
    min_radius = 10
    max_radius = 400
    min_area = 400
    max_area = 50000
    vertical_threshold = 2.1

    # Инициализация GPIO
    GPIO.setmode(GPIO.BCM) # Установка режима нумерации пинов
    servo_pin = 18 # Пин, к которому подключен сервопривод
    GPIO.setup(servo_pin, GPIO.OUT) # Настройка пина как выхода
    pwm = GPIO.PWM(servo_pin, 50) # Создание объекта PWM с частотой 50 Гц
    pwm.start(0) # Запуск PWM с начальной скважностью 0

    # Инициализация двигателей
    motl = pyiArduinoI2Cmotor(0x09) # Левый двигатель
    motl.radius = 12.2 # Установка радиуса колеса
    motr = pyiArduinoI2Cmotor(0x0A) # Правый двигатель
    motr.radius = 12.2 # Установка радиуса колеса

    # Создание объекта детектора цилиндров
    detector = CylinderDetector(
        min_radius=min_radius,
        max_radius=max_radius,
        min_area=min_area,
        max_area=max_area,
        vertical_threshold=vertical_threshold
    )

    # Инициализация камеры
    cap = cv2.VideoCapture(0) # Открытие видеопотока с камеры (индекс 0)
    if not cap.isOpened():
        print("Error: Could not open camera.") # Вывод ошибки, если камера не открылась
        return

    # Настройка параметров камеры
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # Установка ширины кадра
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # Установка высоты кадра
    cap.set(cv2.CAP_PROP_FPS, 30) # Установка частоты кадров

    print("Camera feed started. Press 'q' to exit.") # Сообщение о старте работы

    # Начальное движение робота
    motl.setSpeed(-100, MOT_PWM) 
    motr.setSpeed(100, MOT_PWM)
    time.sleep(3) 

    frame_count = 0 # Счетчик кадров
    start_time = time.time() # Время начала отсчета FPS
    fps = 0 # Переменная для FPS

    # Основной цикл обработки кадров
    while True:
        ret, frame = cap.read() # Чтение кадра с камеры
        if not ret:
            print("Error reading frame.") # Вывод ошибки, если кадр не прочитан
            break

        # Обработка кадра для обнаружения цилиндров
        all_cylinders, masks, center_x = detector.process_frame(frame)
        result = detector.draw_detections(frame, all_cylinders, center_x=center_x)

        # Расчет и отображение FPS
        current_time = time.time()
        elapsed_time = current_time - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        num_cylinders = len(all_cylinders) 
        info_text = f"Cylinders: {num_cylinders} | FPS: {fps:.2f}" 
        cv2.putText(result, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) 

        # Расстояние до центра изображения
        for cyl in all_cylinders:
            px = cyl['area'] 
            distance_from_center = cyl['center'][0] - center_x 

            # сбор цилиндров
            if ck == 2:
                if px > 60000:
                    detector.set_angle(110)
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(2)
                    detector.set_angle(0)
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(3)
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(2)
                    ck -= 1
                elif distance_from_center > 10:
                    motl.setSpeed(100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(1.2)
                elif distance_from_center < -10:
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(0.5)
                elif distance_from_center > -10 and distance_from_center < 10:
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(0.5)

            if cb == 2 and ck == 1:
                if n == 0:
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    n += 1
                if px > 60000:
                    detector.set_angle(110)
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(2)
                    detector.set_angle(0)
                    motl.setSpeed(100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(2)
                    cb -= 1
                elif distance_from_center > 10:
                    motl.setSpeed(100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(1.2)
                elif distance_from_center < -10:
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(0.5)
                elif distance_from_center > -10 and distance_from_center < 10:
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(0.5)

            if cg == 2 and cb == 1 and ck == 1:
                if n == 1:
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(2)
                    n += 1
                if px > 60000:
                    detector.set_angle(110)
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(2)
                    detector.set_angle(0)
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(2)
                    motl.setSpeed(100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(3)
                    cg -= 1
                elif distance_from_center > 10:
                    motl.setSpeed(100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(1.2)
                elif distance_from_center < -10:
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(0.5)
                elif distance_from_center > -10 and distance_from_center < 10:
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(0.5)

            if cw == 2 and cg == 1 and cb == 1 and ck == 1:
                if n == 2:
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(3)
                    n += 1
                elif px > 60000:
                    detector.set_angle(110)
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(3)
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(3)
                    detector.set_angle(0)
                    motl.setSpeed(100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(3)
                    cw -= 1
                if distance_from_center > 10:
                    motl.setSpeed(100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(1.2)
                elif distance_from_center < -10:
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(0.5)
                elif distance_from_center > -10 and distance_from_center < 10:
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(0.5)

            if cw == 1 and cg == 1 and cb == 1 and ck == 1:
                if n == 3:
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(2)
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(5)
                    motl.setSpeed(100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(2)
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(4)
                    motl.setSpeed(100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(3)
                    n += 1
                if px > 60000:
                    detector.set_angle(110)
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(1.2)
                    detector.set_angle(0)
                    ck -= 1
                elif distance_from_center > 10:
                    motl.setSpeed(100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(1.2)
                elif distance_from_center < -10:
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(0.5)
                elif distance_from_center > -10 and distance_from_center < 10:
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(0.5)

            if cw == 1 and cg == 1 and cb == 1 and ck == 0:
                motl.setSpeed(100, MOT_PWM)
                motr.setSpeed(100, MOT_PWM)
                time.sleep(5)
                motl.setSpeed(-100, MOT_PWM)
                motr.setSpeed(100, MOT_PWM)
                time.sleep(3)
                motl.setSpeed(-100, MOT_PWM)
                motr.setSpeed(-100, MOT_PWM)
                time.sleep(4)
                motl.setSpeed(100, MOT_PWM)
                motr.setSpeed(-100, MOT_PWM)
                time.sleep(1)
                motl.setSpeed(-100, MOT_PWM)
                motr.setSpeed(-100, MOT_PWM)
                time.sleep(2)
                motl.setSpeed(100, MOT_PWM)
                motr.setSpeed(-100, MOT_PWM)
                time.sleep(3)
                motl.setSpeed(-100, MOT_PWM)
                motr.setSpeed(-100, MOT_PWM)
                time.sleep(6)
                motl.setSpeed(-100, MOT_PWM)
                motr.setSpeed(100, MOT_PWM)
                time.sleep(1)
                detector.set_angle(0)
                ck = -1

            if cw == 1 and cg == 1 and cb == 1 and ck == -1:
                if n == 4:
                    motl.setSpeed(100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(3)
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(2)
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(1)
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(4)
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(4)
                    n += 1
                if px > 60000:
                    detector.set_angle(110)
                    cg -= 1
                elif distance_from_center > 10:
                    motl.setSpeed(100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(1.2)
                elif distance_from_center < -10:
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(0.5)
                elif distance_from_center > -10 and distance_from_center < 10:
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(0.5)

            if cw == 1 and cg == 0 and cb == 1 and ck == -1:
                if n == 5:
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(5)
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(3)
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(8)
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(2)
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(5)

                    detector.set_angle(0)
                    n += 1
                    cg -= 1
            if cw == 1 and cg == -1 and cb == 1 and ck == -1:
                if n == 6:
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(3)
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(4)
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(1)
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(2)
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(4)
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(17)
                    n += 1
                if px > 60000:
                    detector.set_angle(110)
                    cb -= 1
                elif distance_from_center > 10:
                    motl.setSpeed(100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(1.2)
                elif distance_from_center < -10:
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(0.5)
                elif distance_from_center > -10 and distance_from_center < 10:
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(0.5)
            if cw == 1 and cg == -1 and cb == 0 and ck == -1:
                motl.setSpeed(-100, MOT_PWM)
                motr.setSpeed(-100, MOT_PWM)
                time.sleep(4)
                motl.setSpeed(-100, MOT_PWM)
                motr.setSpeed(100, MOT_PWM)
                time.sleep(3)
                motl.setSpeed(-100, MOT_PWM)
                motr.setSpeed(-100, MOT_PWM)
                time.sleep(5)
                motl.setSpeed(-100, MOT_PWM)
                motr.setSpeed(100, MOT_PWM)
                time.sleep(3)
                motl.setSpeed(-100, MOT_PWM)
                motr.setSpeed(-100, MOT_PWM)
                time.sleep(4)
                detector.set_angle(0)
                motl.setSpeed(-100, MOT_PWM)
                motr.setSpeed(100, MOT_PWM)
                time.sleep(3)
                motl.setSpeed(-100, MOT_PWM)
                motr.setSpeed(-100, MOT_PWM)
                time.sleep(8)
                motl.setSpeed(-100, MOT_PWM)
                motr.setSpeed(100, MOT_PWM)
                time.sleep(3)
                cb = -1
            if cw == 1 and cg == -1 and cb == -1 and ck == -1:
                if px > 60000:
                    detector.set_angle(110)
                    cw -= 1
                elif distance_from_center > 10:
                    motl.setSpeed(100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(1.2)
                elif distance_from_center < -10:
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(0.5)
                elif distance_from_center > -10 and distance_from_center < 10:
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(0.5)
            if cw == 0 and cg == -1 and cb == -1 and ck == -1:
                motl.setSpeed(100, MOT_PWM)
                motr.setSpeed(100, MOT_PWM)
                time.sleep(3)
                motl.setSpeed(-100, MOT_PWM)
                motr.setSpeed(100, MOT_PWM)
                time.sleep(2)
                motl.setSpeed(-100, MOT_PWM)
                motr.setSpeed(-100, MOT_PWM)
                time.sleep(2)
                detector.set_angle(0)
                cw -= 4

        # Отображение результата обработки кадра
        cv2.imshow('Cylinder Detection Result', result)

        frame_count += 1 # Увеличение счетчика кадров

        # Обработка нажатий клавиш
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # Если нажата клавиша 'q', выходим из цикла
            break

    # Освобождение ресурсов
    cap.release() # Освобождение объекта камеры
    cv2.destroyAllWindows() # Закрытие всех окон OpenCV

# Проверка, запущен ли скрипт напрямую
if __name__ == "__main__":
    main() # Вызов основной функции
