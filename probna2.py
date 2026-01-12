import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode = False,
    max_num_hands =2,
    min_detection_confidence = 0.8,
    min_tracking_confidence = 0.5)

cap = cv2.VideoCapture(0)
INDEX_FINGER_TIP_ID = 4
INDEX_FINGER_TIP_ID1 = 8

while cap.isOpened():
    ret , frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape

    image_rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            tip = hand_landmarks.landmark[INDEX_FINGER_TIP_ID]
            tip1 = hand_landmarks.landmark[INDEX_FINGER_TIP_ID1]

            pixel_x = int(tip.x * W)
            pixel_y = int(tip.y * H)
            pixel_x1 = int(tip1.x * W)
            pixel_y1 = int(tip1.y * H)
            cv2.line(frame , (pixel_x , pixel_y) , (pixel_x1, pixel_y1),(0, 255, 0), 3)
            distance = int(((pixel_x - pixel_x1)**2 +(pixel_y - pixel_y1)**2)**0.5)

            cx, cy = (pixel_x + pixel_x1) // 2, (pixel_y + pixel_y1) // 2
            cv2.putText(frame, f"Dist: {distance}", (cx, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.circle(frame, (pixel_x, pixel_y), 15,(255,0,0), -1)
            text = f"X: {pixel_x}, Y: {pixel_y}"
            cv2.circle(frame, (pixel_x1, pixel_y1), 15,(255,0,0), -1)
            text = f"X: {pixel_x1}, Y: {pixel_y1}"

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)
            
    cv2.imshow('Finger Tracking', frame )

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()

