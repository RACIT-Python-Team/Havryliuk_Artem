import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
h = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)  
l = mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4) 

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    c = 0
    text = ""
    if results.multi_hand_landmarks:
        c = len(results.multi_hand_landmarks)
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS, l, h)

            h_img, w_img, _ = frame.shape
            x = int(sum([lm.x for lm in hand.landmark]) / 21 * w_img)
            y = int(sum([lm.y for lm in hand.landmark]) / 21 * h_img)
            text += f'({x},{y}) '
        cv2.putText(frame , f'Hands detected and coord: {len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0} {text}',
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imshow('Hands Detection and coord', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
