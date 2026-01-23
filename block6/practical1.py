
INDEX_FINGER_TIP_ID = 8

import cv2
import mediapipe as mp
import numpy as np


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

INTERACTIVE_ZONES = [
    {"name": "Reception Area", "norm_coords": [0.10, 0.10, 0.45, 0.40], "color": (255, 255, 0)},
    {"name": "Room 1 (Staff)", "norm_coords": [0.55, 0.10, 0.90, 0.40], "color": (0, 165, 255)},
    {"name": "Emergency Exit", "norm_coords": [0.30, 0.60, 0.70, 0.90], "color": (0, 0, 255)}
]
INDEX_FINGER_TIP_ID = 8


def create_virtual_map(W, H):
    map_img = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(1, 10):
        cv2.line(map_img, (0, int(H * i / 10)), (W, int(H * i / 10)), (50, 50, 50), 1)
        cv2.line(map_img, (int(W * i / 10), 0), (int(W * i / 10), H), (50, 50, 50), 1)
    
    cv2.putText(map_img, "VIRTUAL MAP", (W // 2 - 100, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return map_img


cap = cv2.VideoCapture(0)


ROI_WIDTH_PERCENT = 0.7 
ROI_HEIGHT_PERCENT = 0.8
TOUCH_THRESHOLD_Y_PERCENT = 0.55 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    
    ROI_W = int(W * ROI_WIDTH_PERCENT)
    ROI_H = int(H * ROI_HEIGHT_PERCENT)
    ROI_X_START = int((W - ROI_W) / 2)
    ROI_Y_START = int((H - ROI_H) / 2)

    
    TOUCH_THRESHOLD_Y_ABS = int(H * TOUCH_THRESHOLD_Y_PERCENT)

 
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    virtual_map = create_virtual_map(ROI_W, ROI_H)
    
    

    
    for zone in INTERACTIVE_ZONES:
        min_x_pix = int(zone["norm_coords"][0] * ROI_W)
        min_y_pix = int(zone["norm_coords"][1] * ROI_H)
        max_x_pix = int(zone["norm_coords"][2] * ROI_W)
        max_y_pix = int(zone["norm_coords"][3] * ROI_H)
        
        cv2.rectangle(virtual_map, 
                      (min_x_pix, min_y_pix), 
                      (max_x_pix, max_y_pix), 
                      zone["color"], 
                      -1) 
        cv2.putText(virtual_map, zone["name"], (min_x_pix + 10, min_y_pix + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    frame[ROI_Y_START : ROI_Y_START + ROI_H, 
          ROI_X_START : ROI_X_START + ROI_W] = virtual_map
    

    cv2.line(frame, (0, TOUCH_THRESHOLD_Y_ABS), (W, TOUCH_THRESHOLD_Y_ABS), (0, 255, 255), 2)
    cv2.putText(frame, "Y-AXIS TOUCH THRESHOLD", (10, TOUCH_THRESHOLD_Y_ABS - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            tip = hand_landmarks.landmark[INDEX_FINGER_TIP_ID]
            

            abs_x = int(tip.x * W)
            abs_y = int(tip.y * H)

            is_touching_y = abs_y < TOUCH_THRESHOLD_Y_ABS

            is_in_roi = (abs_x >= ROI_X_START and abs_x <= ROI_X_START + ROI_W and
                         abs_y >= ROI_Y_START and abs_y <= ROI_Y_START + ROI_H)

            finger_color = (0, 255, 0)
            
            if is_touching_y and is_in_roi:
                
                relative_x = abs_x - ROI_X_START
                relative_y = abs_y - ROI_Y_START
                
                
                scaled_norm_x = relative_x / ROI_W
                scaled_norm_y = relative_y / ROI_H
                
                finger_color = (0, 255, 255) 
                
           
                
                detected_zone = None
                for zone in INTERACTIVE_ZONES:
                    min_x, min_y, max_x, max_y = zone["norm_coords"]
                    
                   
                    if (scaled_norm_x >= min_x and scaled_norm_x <= max_x and
                        scaled_norm_y >= min_y and scaled_norm_y <= max_y):
                        
                        detected_zone = zone["name"]
                        finger_color = zone["color"] 
                        break #
                
                
                
                if detected_zone:
                    status_text = f"ACTIVE: {detected_zone}"
                 
                    print(f"!!! TOUCH DETECTED: {detected_zone} at ({scaled_norm_x:.2f}, {scaled_norm_y:.2f})")
                else:
                    status_text = f"ACTIVE: Unmapped Area"
                    
                
                cv2.putText(frame, status_text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, finger_color, 2)
                cv2.putText(frame, f"N-Coords: ({scaled_norm_x:.2f}, {scaled_norm_y:.2f})", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, finger_color, 2)

            else:
               
                status_text = "READY / OUTSIDE ZONE"
                cv2.putText(frame, status_text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            
            
            cv2.circle(frame, (abs_x, abs_y), 10, finger_color, -1)


    cv2.imshow('Interactive Map Detection', frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
