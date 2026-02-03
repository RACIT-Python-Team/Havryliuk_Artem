import cv2
import mediapipe as mp
import numpy as np



mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

canvas = None

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

INDEX_FINGER_TIP_ID = 4
INDEX_FINGER_TIP_ID1 = 8

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)  

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            tip = hand_landmarks.landmark[INDEX_FINGER_TIP_ID]
            tip1 = hand_landmarks.landmark[INDEX_FINGER_TIP_ID1]

            x1, y1 = int(tip.x * W), int(tip.y * H)
            x2, y2 = int(tip1.x * W), int(tip1.y * H)

            distance = int(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)

           
            if  200 < distance < 210:
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 3)

           
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(frame, f"Dist: {distance}", (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    frame = cv2.add(frame, canvas)

    cv2.imshow("Finger Tracking", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
