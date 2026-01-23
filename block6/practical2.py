import cv2
import mediapipe as mp
import numpy as np

# Ініціалізація модулів MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

# --- КОНФІГУРАЦІЯ ЗОН ТА ПАРАМЕТРІВ ---
INDEX_FINGER_TIP_ID = 8
INTERACTIVE_ZONES = [
    {"name": "Reception Area", "norm_coords": [0.10, 0.10, 0.45, 0.40], "color": (255, 255, 0)},
    {"name": "Room 1 (Staff)", "norm_coords": [0.55, 0.10, 0.90, 0.40], "color": (0, 165, 255)},
    {"name": "Emergency Exit", "norm_coords": [0.30, 0.60, 0.70, 0.90], "color": (0, 0, 255)}
]
ROI_WIDTH_PERCENT = 0.7 
ROI_HEIGHT_PERCENT = 0.8
TOUCH_THRESHOLD_Y_PERCENT = 0.55 


def create_virtual_map(W, H, zones):
    
    map_img = np.zeros((H, W, 3), dtype=np.uint8)
    
    
    for zone in zones:
        min_x_pix = int(zone["norm_coords"][0] * W)
        min_y_pix = int(zone["norm_coords"][1] * H)
        max_x_pix = int(zone["norm_coords"][2] * W)
        max_y_pix = int(zone["norm_coords"][3] * H)
        
        
        cv2.rectangle(map_img, (min_x_pix, min_y_pix), (max_x_pix, max_y_pix), zone["color"], -1) 
        
        cv2.putText(map_img, zone["name"], (min_x_pix + 10, min_y_pix + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return map_img

cap = cv2.VideoCapture(0)

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
    virtual_map = create_virtual_map(ROI_W, ROI_H, INTERACTIVE_ZONES)
    frame[ROI_Y_START : ROI_Y_START + ROI_H, ROI_X_START : ROI_X_START + ROI_W] = virtual_map
    
    
    cv2.line(frame, (0, TOUCH_THRESHOLD_Y_ABS), (W, TOUCH_THRESHOLD_Y_ABS), (0, 255, 255), 2)


    
   
   
    
    detected_zone = None
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            tip = hand_landmarks.landmark[INDEX_FINGER_TIP_ID]
            
            abs_x = int(tip.x * W)
            abs_y = int(tip.y * H)
            
            # 1. Перевірка Умов Активації
            is_touching_y = abs_y < TOUCH_THRESHOLD_Y_ABS
            is_in_roi = (abs_x >= ROI_X_START and abs_x <= ROI_X_START + ROI_W and
                         abs_y >= ROI_Y_START and abs_y <= ROI_Y_START + ROI_H)
            
            finger_color = (0, 255, 0)
            
            if is_touching_y and is_in_roi:
                
                
                relative_x = abs_x - ROI_X_START
                relative_y = abs_y - ROI_Y_START
                scaled_norm_x = relative_x / ROI_W
                scaled_norm_y = relative_y / ROI_H
                
                finger_color = (255, 255, 255) 
                
                
                for zone in INTERACTIVE_ZONES:
                    min_x, min_y, max_x, max_y = zone["norm_coords"]
                    
                    
                    if (scaled_norm_x >= min_x and scaled_norm_x <= max_x and
                        scaled_norm_y >= min_y and scaled_norm_y <= max_y):
                        
                        detected_zone = zone["name"]
                        finger_color = zone["color"] 
                        
                        zone_min_x = ROI_X_START + int(min_x * ROI_W)
                        zone_min_y = ROI_Y_START + int(min_y * ROI_H)
                        zone_max_x = ROI_X_START + int(max_x * ROI_W)
                        zone_max_y = ROI_Y_START + int(max_y * ROI_H)
                        
                        cv2.rectangle(frame, 
                                      (zone_min_x, zone_min_y), 
                                      (zone_max_x, zone_max_y), 
                                      (255, 255, 255), 4) 
                        
                        break 
                
                if detected_zone:
                    status_text = f"ACTIVE: {detected_zone}"
                else:
                    status_text = f"ACTIVE: Unmapped Area"
                
                
                cv2.putText(frame, f"N-Coords: ({scaled_norm_x:.2f}, {scaled_norm_y:.2f})", 
                            (abs_x + 10, abs_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, finger_color, 2)
            
            
            cv2.putText(frame, status_text if detected_zone else "Awaiting Touch", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, finger_color, 2)

          
            cv2.circle(frame, (abs_x, abs_y), 10, finger_color, -1)


    cv2.imshow('Zone Hit Test', frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()