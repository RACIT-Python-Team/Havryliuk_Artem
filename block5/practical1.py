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

is_touching = True

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1) 
    H, W, _ = frame.shape      
    TOUCH_THRESHOLD_Y = int(H * 0.60) 

    
    cv2.line(frame, (0, TOUCH_THRESHOLD_Y), (W, TOUCH_THRESHOLD_Y), (255, 255, 0), 2)
    cv2.putText(frame, "TOUCH SURFACE (Y = 60%)", (10, TOUCH_THRESHOLD_Y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
           
            tip = hand_landmarks.landmark[INDEX_FINGER_TIP_ID]
            
    
            pixel_x = int(tip.x * W)
            pixel_y = int(tip.y * H)
            
           
            cv2.circle(frame, (pixel_x, pixel_y), 10, (0, 165, 255), -1) 
            
      
            if pixel_y < TOUCH_THRESHOLD_Y:
                
                if is_touching:
                   
                    is_touching = True
                    print(f"!!! START TOUCH !!! X: {pixel_x}, Y: {pixel_y}")
                    
                color = (0, 0, 255) 
                touch_status = "TOUCHING"
            else:
             
                is_touching = False
                color = (0, 255, 0) 
                touch_status = "READY"

       
            cv2.putText(frame, touch_status, (pixel_x + 20, pixel_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.circle(frame, (pixel_x, pixel_y), 15, color, 2) 

    cv2.imshow('Touch Detection by Y-Coordinate', frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()