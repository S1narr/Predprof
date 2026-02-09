# Импортируем необходимые библиотеки
import cv2                      # Библиотека для компьютерного зрения (обработка изображений)
import numpy as np                # Библиотека для численных операций (работа с массивами)
import math                       # Математическая библиотека
from pyiArduinoI2Cmotor import *  # Библиотека для управления моторами Arduino по I2C
import time                       # Библиотека для работы со временем (задержки)
import RPi.GPIO as GPIO           # Библиотека для управления GPIO на Raspberry Pi


# Класс для обнаружения цилиндров
class CylinderDetector:
    # Конструктор класса
    def __init__(self, min_radius=10, max_radius=400, min_area=500, max_area=50000, vertical_threshold=2.0):
        # Инициализация параметров обнаружения цилиндров
        self.min_radius = min_radius          # Минимальный радиус цилиндра для обнаружения
        self.max_radius = max_radius          # Максимальный радиус цилиндра для обнаружения
        self.min_area = min_area              # Минимальная площадь контура цилиндра
        self.max_area = max_area              # Максимальная площадь контура цилиндра
        self.vertical_threshold = vertical_threshold # Порог для определения вертикального цилиндра (отношение высоты к ширине)

        # Создание структурного элемента для морфологических операций
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Определение цветов для отрисовки
        self.green = (0, 255, 0)  # Зеленый
        self.red = (0, 0, 255)    # Красный
        self.blue = (255, 0, 0)   # Синий

        # Определение диапазонов цветов в HSV для масок
        self.color_ranges = {
            'red': ([0, 50, 50], [10, 255, 255]),      # Диапазон для красного цвета
            'green': ([35, 50, 50], [85, 255, 255]),    # Диапазон для зеленого цвета
            'blue': ([100, 50, 50], [140, 255, 255]),   # Диапазон для синего цвета
            'white': ([0, 0, 205], [180, 30, 255])     # Диапазон для белого цвета
        }
        # Цвета для отрисовки обнаруженных объектов
        self.color_draw = {'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0), 'white': (255, 255, 255)}

    # Функция предобработки изображения для выделения определенного цвета
    def preprocess_image(self, image, color):
        try:
            # Преобразование изображения из BGR в HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Установка нижнего и верхнего порогов для заданного цвета
            lower_color = np.array(self.color_ranges[color][0])
            upper_color = np.array(self.color_ranges[color][1])

            # Создание маски для заданного цвета
            mask = cv2.inRange(hsv, lower_color, upper_color)

            # Дополнительные маски для красного цвета (учитывает перекрытие цветового круга)
            if color == 'red':
                lower_color2 = np.array([170, 50, 50])
                upper_color2 = np.array([180, 255, 255])
                mask2 = cv2.inRange(hsv, lower_color2, upper_color2)
                mask = cv2.bitwise_or(mask, mask2) # Объединение масок

                lower_color3 = np.array([150, 50, 50])
                upper_color3 = np.array([170, 255, 255])
                mask3 = cv2.inRange(hsv, lower_color3, upper_color3)
                mask = cv2.bitwise_or(mask, mask3) # Объединение масок

            # Дополнительные маски для зеленого цвета (для более точного обнаружения)
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

            # Дополнительные маски для синего цвета
            elif color == 'blue':
                lower_blue2 = np.array([140, 50, 50])
                upper_blue2 = np.array([160, 255, 255])
                mask_blue2 = cv2.inRange(hsv, lower_blue2, upper_blue2)
                mask = cv2.bitwise_or(mask, mask_blue2)

            # Применение морфологических операций для очистки маски
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)  # Открытие (удаление мелких шумов)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel) # Закрытие (заполнение маленьких дыр)

            return mask, hsv # Возвращаем очищенную маску и изображение в HSV
        except Exception as e:
            print(f"Error during image preprocessing for color {color}: {e}") # Выводим сообщение об ошибке
            return None, None # Возвращаем None в случае ошибки

    # Функция обнаружения кругов с помощью преобразования Хафа
    def detect_circles_hough(self, mask):
        try:
            # Применение преобразования Хафа для поиска кругов
            circles = cv2.HoughCircles(
                mask,
                cv2.HOUGH_GRADIENT, # Алгоритм обнаружения
                dp=10,              # Отношение инверсного разрешения детектора к исходному
                minDist=100,        # Минимальное расстояние между центрами обнаруженных кругов
                param1=100,         # Верхний порог для детектора градиентов Canny
                param2=100,         # Порог для центра окружности в процессе обнаружения
                minRadius=self.min_radius, # Минимальный радиус круга
                maxRadius=self.max_radius  # Максимальный радиус круга
            )
            return circles # Возвращаем найденные круги
        except Exception as e:
            print(f"Error during Hough Circle Transform detection: {e}") # Выводим сообщение об ошибке
            return None # Возвращаем None в случае ошибки

    # Функция обнаружения контуров и фильтрации их до цилиндров
    def detect_contours(self, mask, color):
        # Поиск всех контуров на маске
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cylinders = [] # Список для хранения обнаруженных цилиндров
        for contour in contours:
            area = cv2.contourArea(contour) # Вычисление площади контура
            # Проверка, находится ли площадь контура в допустимых пределах
            if self.min_area < area < self.max_area:

                # Аппроксимация контура многоугольником
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Получение ограничивающего прямоугольника контура
                x, y, w, h = cv2.boundingRect(contour)

                aspect_ratio = float(w) / h # Вычисление соотношения сторон
                # Проверка, находится ли соотношение сторон в допустимых пределах (для цилиндра 0.3 < aspect_ratio < 2.0)
                if 0.3 < aspect_ratio < 2.0:

                    # Проверка, является ли цилиндр вертикальным (высота больше ширины и отношение высоты к ширине больше порога)
                    if h > w and h / w > self.vertical_threshold:

                        # Нахождение минимального описывающего круга
                        (cx, cy), radius = cv2.minEnclosingCircle(contour)

                        circle_area = math.pi * radius * radius # Вычисление площади описывающего круга
                        fill_ratio = area / circle_area if circle_area > 0 else 0 # Вычисление коэффициента заполнения

                        # Проверка, находится ли коэффициент заполнения в допустимых пределах (для цилиндра 0.3 < fill_ratio < 0.9)
                        if 0.3 < fill_ratio < 0.9:
                            # Добавление информации об обнаруженном цилиндре в список
                            cylinders.append({
                                'contour': contour,
                                'center': (int(cx), int(cy)),
                                'radius': int(radius),
                                'area': area,
                                'bbox': (x, y, w, h),
                                'color': color
                            })

        return cylinders # Возвращаем список обнаруженных цилиндров

    # Основная функция обработки кадра
    def process_frame(self, frame):
        all_cylinders = [] # Список для всех обнаруженных цилиндров
        masks = {}           # Словарь для хранения масок разных цветов

        height, width = frame.shape[:2] # Получение размеров кадра
        center_x = width // 2           # Вычисление координаты центра кадра по оси X

        # Перебор всех заданных цветов для обнаружения
        for color in self.color_ranges.keys():
            mask, hsv = self.preprocess_image(frame, color) # Предобработка изображения для текущего цвета

            if mask is None or hsv is None: # Если предобработка не удалась, пропускаем
                continue

            cylinders = self.detect_contours(mask, color) # Обнаружение контуров (цилиндров)
            all_cylinders.extend(cylinders)               # Добавление найденных цилиндров в общий список

            masks[color] = mask # Сохранение маски для текущего цвета

        return all_cylinders, masks, center_x # Возвращаем все цилиндры, маски и координату центра

    # Метод для установки угла сервопривода
    def set_angle(self,angle):
        """Установить угол серво (0–180 градусов)"""
        duty = (angle / 18) + 2 # Расчет скважности ШИМ для заданного угла
        pwm.ChangeDutyCycle(duty) # Установка скважности
        time.sleep(0.5) # Небольшая задержка для стабилизации

# Основная функция программы
def main():
    # Инициализация переменных состояния для управления последовательностью действий
    cb=2
    ck=2
    cg=2
    cw=2
    n=0 # счетчик движений

    # Параметры обнаружения цилиндров
    min_radius = 10
    max_radius = 400
    min_area = 400
    max_area = 50000
    vertical_threshold = 1.5 # Исправлено значение для более точного определения вертикальности

    # Настройка GPIO
    GPIO.setmode(GPIO.BCM) # Установка режима нумерации пинов BCM
    servo_pin = 18           # Пин, к которому подключен сервопривод
    GPIO.setup(servo_pin, GPIO.OUT) # Настройка пина сервопривода как выход
    pwm = GPIO.PWM(servo_pin, 50)   # Создание объекта PWM для управления сервоприводом (частота 50 Гц)
    pwm.start(0)            # Запуск PWM с начальной скважностью 0

    # Инициализация моторов
    motl = pyiArduinoI2Cmotor(0x09) # Инициализация левого мотора
    motl.radius = 12.2              # Установка радиуса колеса левого мотора
    motr = pyiArduinoI2Cmotor(0x0A) # Инициализация правого мотора
    motr.radius = 12.2              # Установка радиуса колеса правого мотора

    # Создание объекта детектора цилиндров с заданными параметрами
    detector = CylinderDetector(
        min_radius=min_radius,
        max_radius=max_radius,
        min_area=min_area,
        max_area=max_area,
        vertical_threshold=vertical_threshold
    )

    # Инициализация камеры
    cap = cv2.VideoCapture(0) # Открытие камеры (индекс 0 для стандартной веб-камеры)
    if not cap.isOpened():    # Проверка, удалось ли открыть камеру
        print("Error: Could not open camera.") # Вывод сообщения об ошибке
        return # Выход из функции, если камера не открылась

    # Установка параметров видеопотока
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Установка ширины кадра
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # Установка высоты кадра
    cap.set(cv2.CAP_PROP_FPS, 30)           # Установка частоты кадров

    print("Camera feed started. Press 'q' to exit.") # Сообщение о начале работы

    # Начальное движение робота
    motl.setSpeed(-100, MOT_PWM) # Установка скорости левого мотора (назад)
    motr.setSpeed(100, MOT_PWM)  # Установка скорости правого мотора (вперед)
    time.sleep(2)                # Задержка 2 секунды

    # Основной цикл обработки кадров
    while True:
        ret, frame = cap.read() # Чтение кадра с камеры
        if not ret:             # Если кадр не удалось прочитать
            print("Error reading frame.") # Вывод сообщения об ошибке
            break               # Выход из цикла

        # Обработка кадра для обнаружения цилиндров
        all_cylinders, masks, center_x = detector.process_frame(frame)

        # Обработка каждого обнаруженного цилиндра
        for cyl in all_cylinders:
            px = cyl['area']                                         # Площадь цилиндра в пикселях
            distance_from_center = cyl['center'][0] - center_x       # Расстояние от центра цилиндра до центра кадра
            print(f"Цилиндр {cyl['color']}: Центр ({cyl['center'][0]}, {cyl['center'][1]}), "
                  f"Площадь: {px} пикселей, Расстояние от центра: {distance_from_center} px") # Вывод информации о цилиндре

            # Логика управления роботом на основе обнаруженных цилиндров и состояния
            # Это очень сложный и запутанный блок кода, который реализует последовательность действий.
            # Каждый if-блок соответствует определенному этапу выполнения задачи.
            # Переменные cb, ck, cg, cw, n используются для отслеживания текущего этапа и выполненных действий.

            # Начальный этап: поиск и захват первого цилиндра (предположительно черного)
            if ck==2: # Если еще не захвачен первый цилиндр
                if px > 60000: # Если цилиндр достаточно большой (близко)
                    detector.set_angle(110) # Поворот сервопривода для захвата
                    motl.setSpeed(-100, MOT_PWM) # Движение назад
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(1.2) # Задержка для выполнения движения
                    detector.set_angle(0) # Возврат сервопривода в исходное положение
                    ck-=1 # Переход к следующему этапу
                elif distance_from_center > 10: # Если цилиндр справа от центра
                    motl.setSpeed(100, MOT_PWM) # Поворот вправо
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(1.2)
                elif distance_from_center < -10: # Если цилиндр слева от центра
                    motl.setSpeed(-100, MOT_PWM) # Поворот влево
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(0.5)
                elif distance_from_center > -10 and distance_from_center < 10: # Если цилиндр по центру
                    motl.setSpeed(-100, MOT_PWM) # Движение вперед
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(0.5)

            # Этап 2: захват второго цилиндра (предположительно синего)
            if cb==2 and ck==1: # Если еще не захвачен второй цилиндр и первый уже захвачен
                if n==0: # Начало движения для поиска второго цилиндра
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(3)
                    n+=1 # Счетчик движения
                if px > 60000: # Если цилиндр большой (близко)
                    detector.set_angle(110) # Поворот сервопривода
                    motl.setSpeed(-100, MOT_PWM) # Движение назад
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(1.2)
                    detector.set_angle(0) # Возврат сервопривода
                    cb-=1 # Переход к следующему этапу
                elif distance_from_center > 10: # Если цилиндр справа
                    motl.setSpeed(100, MOT_PWM) # Поворот вправо
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(1.2)
                elif distance_from_center < -10: # Если цилиндр слева
                    motl.setSpeed(-100, MOT_PWM) # Поворот влево
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(0.5)
                elif distance_from_center > -10 and distance_from_center < 10: # Если цилиндр по центру
                    motl.setSpeed(-100, MOT_PWM) # Движение вперед
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(0.5)

            # Этап 3: захват третьего цилиндра (предположительно зеленого)
            if cg==2 and cb==1 and ck==1: # Если еще не захвачен третий цилиндр
                if n==1: # Начало движения для поиска третьего цилиндра
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(3)
                    n+=1
                if px > 60000: # Если цилиндр большой (близко)
                    detector.set_angle(110) # Поворот сервопривода
                    motl.setSpeed(-100, MOT_PWM) # Движение назад
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(1.2)
                    detector.set_angle(0) # Возврат сервопривода
                    cg-=1 # Переход к следующему этапу
                elif distance_from_center > 10: # Если цилиндр справа
                    motl.setSpeed(100, MOT_PWM) # Поворот вправо
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(1.2)
                elif distance_from_center < -10: # Если цилиндр слева
                    motl.setSpeed(-100, MOT_PWM) # Поворот влево
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(0.5)
                elif distance_from_center > -10 and distance_from_center < 10: # Если цилиндр по центру
                    motl.setSpeed(-100, MOT_PWM) # Движение вперед
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(0.5)

            # Этап 4: захват четвертого цилиндра (предположительно белого)
            if cw==2 and cg==1 and cb==1 and ck==1: # Если еще не захвачен четвертый цилиндр
                if n==2: # Начало движения для поиска четвертого цилиндра
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(3)
                    n+=1
                elif px > 60000: # Если цилиндр большой (близко)
                    detector.set_angle(110) # Поворот сервопривода
                    motl.setSpeed(-100, MOT_PWM) # Движение назад
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(1.2)
                    detector.set_angle(0) # Возврат сервопривода
                    cw-=1 # Переход к следующему этапу
                if distance_from_center > 10: # Если цилиндр справа
                    motl.setSpeed(100, MOT_PWM) # Поворот вправо
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(1.2)
                elif distance_from_center < -10: # Если цилиндр слева
                    motl.setSpeed(-100, MOT_PWM) # Поворот влево
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(0.5)
                elif distance_from_center > -10 and distance_from_center < 10: # Если цилиндр по центру
                    motl.setSpeed(-100, MOT_PWM) # Движение вперед
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(0.5)

            # Этап 5: сбор всех цилиндров в одну кучу
            if cw==1 and cg==1 and cb==1 and ck==1:
                if n==3: # Начало выполнения последовательности движений
                    motl.setSpeed(-100, MOT_PWM) # Поворот
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(2)
                    motl.setSpeed(-100, MOT_PWM) # Движение вперед
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(5)
                    motl.setSpeed(100, MOT_PWM) # Поворот
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(2)
                    motl.setSpeed(-100, MOT_PWM) # Движение вперед
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(4)
                    motl.setSpeed(100, MOT_PWM) # Поворот
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(3)
                    n += 1
                if px>60000 : # Если обнаружен цилиндр (для коррекции положения)
                    detector.set_angle(110) # Поворот сервопривода
                    motl.setSpeed(-100, MOT_PWM) # Движение назад
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(1.2)
                    detector.set_angle(0) # Возврат сервопривода
                    ck-=1 # Уменьшаем счетчик ck, чтобы выйти из этого блока
                elif distance_from_center > 10: # Поворот вправо
                    motl.setSpeed(100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(1.2)
                elif distance_from_center < -10: # Поворот влево
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(0.5)
                elif distance_from_center > -10 and distance_from_center < 10: # Движение вперед
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(0.5)

            # Завершающий этап: сбор всех цилиндров и перемещение
            if cw == 1 and cg == 1 and cb == 1 and ck == 0: # Если все цилиндры были "захвачены" (состояние ck стало 0)
                motl.setSpeed(100, MOT_PWM) # Движение вперед
                motr.setSpeed(100, MOT_PWM)
                time.sleep(5)
                motl.setSpeed(-100, MOT_PWM) # Поворот
                motr.setSpeed(100, MOT_PWM)
                time.sleep(3)
                motl.setSpeed(-100, MOT_PWM) # Движение вперед
                motr.setSpeed(-100, MOT_PWM)
                time.sleep(4)
                motl.setSpeed(100, MOT_PWM) # Поворот
                motr.setSpeed(-100, MOT_PWM)
                time.sleep(1)
                motl.setSpeed(-100, MOT_PWM) # Движение вперед
                motr.setSpeed(-100, MOT_PWM)
                time.sleep(2)
                motl.setSpeed(100, MOT_PWM) # Поворот
                motr.setSpeed(-100, MOT_PWM)
                time.sleep(3)
                motl.setSpeed(-100, MOT_PWM) # Движение вперед
                motr.setSpeed(-100, MOT_PWM)
                time.sleep(6)
                motl.setSpeed(-100, MOT_PWM) # Поворот
                motr.setSpeed(100, MOT_PWM)
                time.sleep(1)
                detector.set_angle(0) # Возврат сервопривода
                ck=-1 # Устанавливаем состояние, чтобы этот блок не выполнялся повторно

            # Этап 7: последовательность действий после завершения сбора
            if cw == 1 and cg == 1 and cb == 1 and ck == -1:
                if n==4: # Начало последовательности движений
                    motl.setSpeed(100, MOT_PWM) # Поворот
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(3)
                    motl.setSpeed(-100, MOT_PWM) # Движение вперед
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(7)
                    motl.setSpeed(-100, MOT_PWM) # Поворот
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(3)
                    n+=1
                if px>60000 : # Коррекция положения по обнаруженному цилиндру
                    detector.set_angle(110) # Поворот сервопривода
                    cg-=1 # Переход к следующему этапу (снижаем счетчик cg)
                elif distance_from_center > 10: # Поворот вправо
                    motl.setSpeed(100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(1.2)
                elif distance_from_center < -10: # Поворот влево
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(0.5)
                elif distance_from_center > -10 and distance_from_center < 10: # Движение вперед
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(0.5)

            # Этап 8: дальнейшая последовательность движений
            if cw == 1 and cg == 0 and cb == 1 and ck == -1:
                if n==5: # Начало последовательности движений
                    motl.setSpeed(-100, MOT_PWM) # Движение вперед
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(5)
                    motl.setSpeed(-100, MOT_PWM) # Поворот
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(3)
                    motl.setSpeed(-100, MOT_PWM) # Движение вперед
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(8)
                    motl.setSpeed(-100, MOT_PWM) # Поворот
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(2)
                    motl.setSpeed(-100, MOT_PWM) # Движение вперед
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(2)
                    detector.set_angle(0) # Возврат сервопривода
                    n+=1
                    cg-=1 # Переход к следующему этапу

            # Этап 9: еще одна последовательность движений
            if cw == 1 and cg == -1 and cb == 1 and ck == -1:
                if n==6: # Начало последовательности движений
                    motl.setSpeed(-100, MOT_PWM) # Поворот
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(2)
                    motl.setSpeed(-100, MOT_PWM) # Движение вперед
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(6)
                    motl.setSpeed(-100, MOT_PWM) # Поворот
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(3)
                    n+=1
                if px>60000 : # Коррекция положения
                    detector.set_angle(110) # Поворот сервопривода
                    cb-=1 # Переход к следующему этапу
                elif distance_from_center > 10: # Поворот вправо
                    motl.setSpeed(100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(1.2)
                elif distance_from_center < -10: # Поворот влево
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(0.5)
                elif distance_from_center > -10 and distance_from_center < 10: # Движение вперед
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(0.5)

            # Этап 10: Завершающая последовательность движения
            if cw == 1 and cg == -1 and cb == 0 and ck == -1:
                motl.setSpeed(-100, MOT_PWM) # Движение вперед
                motr.setSpeed(-100, MOT_PWM)
                time.sleep(3)
                motl.setSpeed(-100, MOT_PWM) # Поворот
                motr.setSpeed(100, MOT_PWM)
                time.sleep(3)
                motl.setSpeed(-100, MOT_PWM) # Движение вперед
                motr.setSpeed(-100, MOT_PWM)
                time.sleep(12)
                motl.setSpeed(-100, MOT_PWM) # Поворот
                motr.setSpeed(100, MOT_PWM)
                time.sleep(3)
                motl.setSpeed(-100, MOT_PWM) # Движение вперед
                motr.setSpeed(-100, MOT_PWM)
                time.sleep(2)
                detector.set_angle(0) # Возврат сервопривода
                cb=-1 # Уменьшаем счетчик cb

            # Этап 11: Финальная задача
            if cw == 1 and cg == -1 and cb == -1 and ck == -1:
                if n==7: # Начало финальной последовательности
                    motl.setSpeed(100, MOT_PWM) # Поворот
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(3)
                    motl.setSpeed(-100, MOT_PWM) # Движение вперед
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(8)
                    motl.setSpeed(-100, MOT_PWM) # Поворот
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(4)
                    motl.setSpeed(-100, MOT_PWM) # Поворот
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(4)
                    n+=1
                if px>60000 : # Коррекция положения
                    detector.set_angle(110) # Поворот сервопривода
                    cw-=1 # Переход к финальному этапу
                elif distance_from_center > 10: # Поворот вправо
                    motl.setSpeed(100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(1.2)
                elif distance_from_center < -10: # Поворот влево
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(100, MOT_PWM)
                    time.sleep(0.5)
                elif distance_from_center > -10 and distance_from_center < 10: # Движение вперед
                    motl.setSpeed(-100, MOT_PWM)
                    motr.setSpeed(-100, MOT_PWM)
                    time.sleep(0.5)

            # Финальный блок: после выполнения всех задач
            if cw == 0 and cg == -1 and cb == -1 and ck == -1:
                motl.setSpeed(100, MOT_PWM) # Движение вперед
                motr.setSpeed(100, MOT_PWM)
                time.sleep(3)
                motl.setSpeed(-100, MOT_PWM) # Поворот
                motr.setSpeed(100, MOT_PWM)
                time.sleep(2)
                motl.setSpeed(-100, MOT_PWM) # Движение вперед
                motr.setSpeed(-100, MOT_PWM)
                time.sleep(2)
                detector.set_angle(0) # Возврат сервопривода

        # Отображение видеопотока с камеры
        cv2.imshow('Camera Feed', frame)

        # Обработка нажатий клавиш


    # Освобождение ресурсов
    cap.release()       # Освобождение объекта камеры
    cv2.destroyAllWindows() # Закрытие всех окон OpenCV

# Проверка, является ли скрипт основным исполняемым файлом
if __name__ == "__main__":
    main() # Вызов основной функции