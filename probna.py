
import cv2
import numpy as np

# Глобальна змінна для збереження кадру
frame = None

# Функція зворотного виклику для обробки подій миші
def get_hsv_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Конвертуємо кадр у HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Отримуємо значення пікселя в HSV
        hsv_value = hsv_frame[y, x]
        
        print(f"BGR колір: {frame[y, x]}")
        print(f"HSV колір: {hsv_value}")
        print("-" * 30)

# Ініціалізація вікна та прив'язка функції до події миші
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', get_hsv_color)

# Завантаження зображення або відеопотоку
cap = cv2.VideoCapture(0) # 0 - для веб-камери

print("Натисніть ліву кнопку миші на об'єкті для отримання його HSV-значень.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('Image', frame)
    
    # Вихід при натисканні 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()