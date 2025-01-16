import cv2
import pandas as pd
import numpy as np

color_data = pd.read_csv('colornames.csv')

def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
color_data[['r', 'g', 'b']] = pd.DataFrame(color_data['hex'].apply(lambda x: hex_to_rgb(x)).tolist(), index=color_data.index)

def closest_color(r, g, b):
    min_dist = float('inf')
    closest_name = None
    for _, row in color_data.iterrows():
        dr = (r - row['r']) ** 2
        dg = (g - row['g']) ** 2
        db = (b - row['b']) ** 2
        distance = np.sqrt(dr + dg + db)

        if distance < min_dist:
            min_dist = distance
            closest_name = row['color_name']
    return closest_name

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Tidak dapat membuka kamera")
    exit()
frame_size = 100

while True:
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat membaca frame dari kamera")
        break

    height, width, _ = frame.shape
    start_x = width // 2 - frame_size // 2
    start_y = height // 2 - frame_size // 2
    end_x = start_x + frame_size
    end_y = start_y + frame_size

    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    cropped_frame = frame[start_y:end_y, start_x:end_x]

    center = cropped_frame[frame_size // 2, frame_size // 2]
    r, g, b = int(center[2]), int(center[1]), int(center[0])

    warna_terdekat = closest_color(r, g, b)
    text_position = (10, 30)

    text_size, _ = cv2.getTextSize(f"Warna: {warna_terdekat}", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    text_width, text_height = text_size
    box_x1 = text_position[0] - 10
    box_y1 = text_position[1] - text_height - 10
    box_x2 = box_x1 + text_width + 20
    box_y2 = box_y1 + text_height + 20

    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)  # hitam

    cv2.putText(frame, f"Warna: {warna_terdekat}", text_position, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imshow("Deteksi Warna", frame)

    if cv2.waitKey(1) & 0xFF == ord('w'): #tekan w untuk keluar dari kamera
        break

cap.release()
cv2.destroyAllWindows()