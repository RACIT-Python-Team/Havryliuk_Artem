import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

INDEX_FINGER_TIP_ID = 8
cap = cv2.VideoCapture(0)


ROI_W = 400  
ROI_H = 400 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    
    ROI_X_START = int((W - ROI_W) / 2)
    ROI_Y_START = int((H - ROI_H) / 2)
    TOUCH_THRESHOLD_Y = int((H -ROI_Y_START) * 0.50) 
   
    cv2.line(frame, (0, TOUCH_THRESHOLD_Y), (W, TOUCH_THRESHOLD_Y), (255, 255, 0), 2)
    cv2.rectangle(frame, 
                  (ROI_X_START, ROI_Y_START), 
                  (ROI_X_START + ROI_W, ROI_Y_START + ROI_H), 
                  (255, 0, 0), 3)
    cv2.putText(frame, "VIRTUAL MAP AREA (ROI)", (ROI_X_START, ROI_Y_START - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            tip = hand_landmarks.landmark[INDEX_FINGER_TIP_ID]
            
            
            abs_x = int(tip.x * W)
            abs_y = int(tip.y * H)
            

            if (abs_x >= ROI_X_START and abs_x <= ROI_X_START + ROI_W and
                abs_y >= ROI_Y_START and abs_y <= ROI_Y_START + ROI_H):

                status_color = (0, 255, 0)
                relative_x = abs_x - ROI_X_START
                relative_y = abs_y - ROI_Y_START
                
   
                scaled_norm_x = relative_x / ROI_W
                scaled_norm_y = relative_y / ROI_H
                
                
                status_color = (0, 255, 0)
               
                text2 = f"MAP NORM: ({scaled_norm_x:.2f}, {scaled_norm_y:.2f})"
                
                
                cv2.circle(frame, (abs_x, abs_y), 10, status_color, -1)
                
                cv2.putText(frame, text2, (abs_x + 10, abs_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
                
            else:
                
                status_color = (0, 165, 255)
                cv2.circle(frame, (abs_x, abs_y), 5, status_color, -1)
                cv2.putText(frame, "OUTSIDE MAP", (abs_x + 10, abs_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

            if (abs_y > TOUCH_THRESHOLD_Y):
                print(f"TOUCH DETECTED at MAP NORM: (X: {scaled_norm_x:.2f}, Y: {scaled_norm_y:.2f})")

    cv2.imshow('Scaled Coordinates for Map', frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()