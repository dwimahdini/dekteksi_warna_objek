import cv2
import numpy as np
import pandas as pd

# Membaca dataset warna dalam format HSV
df = pd.read_csv('colornames_hsv.csv')

# Inisialisasi dictionary untuk menyimpan range warna
color_ranges = {}

for _, row in df.iterrows():
    lower = np.array([row['lower_h'], row['lower_s'], row['lower_v']], dtype=np.uint8)
    upper = np.array([row['upper_h'], row['upper_s'], row['upper_v']], dtype=np.uint8)
    color_ranges[row['color_name']] = {'lower': lower, 'upper': upper}

# Memulai capture video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    for color_name, ranges in color_ranges.items():
        mask = cv2.inRange(hsv_frame, ranges['lower'], ranges['upper'])
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Color Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
