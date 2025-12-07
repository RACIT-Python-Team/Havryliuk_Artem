import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    low = np.array([58, 50 , 100])
    up = np.array([73, 108 , 168])
    
    mask = cv2.inRange(hsv, low, up)
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > 500:

            M = cv2.moments(largest_contour)
            
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                cv2.drawContours(frame, [largest_contour], -1, (255, 0, 0), 2)
                
                cv2.circle(frame, (cx, cy), 7, (100, 0, 255), -1)
                
                coord_text = f"X: {cx}, Y: {cy}"
                cv2.putText(frame, coord_text, (cx - 20, cy - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Video Direction", frame)
    cv2.imshow("Mask (Black & White)", mask) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()