import cv2
import numpy as np
import pandas as pd

def create_color_range(hsv, name):
    """Membuat range HSV yang sesuai untuk setiap warna"""
    h, s, v = hsv
    
    if name in ['Putih']:
        return {
            'lower': np.array([0, 0, 200], dtype=np.uint8),
            'upper': np.array([180, 30, 255], dtype=np.uint8)
        }
    elif name in ['Hitam']:
        return {
            'lower': np.array([0, 0, 0], dtype=np.uint8),
            'upper': np.array([180, 255, 30], dtype=np.uint8)
        }
    elif name in ['Abu-abu', 'Hitam Muda']:
        return {
            'lower': np.array([0, 0, 40], dtype=np.uint8),
            'upper': np.array([180, 30, 220], dtype=np.uint8)
        }
    elif name in ['Coklat', 'Merah Marun']:
        return {
            'lower1': np.array([0, 100, 100], dtype=np.uint8),
            'upper1': np.array([10, 255, 255], dtype=np.uint8),
            'lower2': np.array([170, 100, 100], dtype=np.uint8),
            'upper2': np.array([180, 255, 255], dtype=np.uint8)
        }
    else:
        return {
            'lower': np.array([max(0, h-10), 100, 100], dtype=np.uint8),
            'upper': np.array([min(180, h+10), 255, 255], dtype=np.uint8)
        }

def put_text_with_background(img, text, position, font_scale=0.7):
    """Fungsi untuk menampilkan teks dengan background"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Mendapatkan ukuran teks
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, 2)
    
    # Menambah padding
    padding_x = 10
    padding_y = 5
    
    # Koordinat untuk background
    x, y = position
    background_coords = [
        (x - padding_x, y - text_height - padding_y),
        (x + text_width + padding_x, y + padding_y)
    ]
    
    # Menggambar background
    cv2.rectangle(img, 
                 background_coords[0],  # top-left
                 background_coords[1],  # bottom-right
                 (0, 0, 0),            # warna hitam
                 -1)                   # fill rectangle
    
    # Menggambar teks
    cv2.putText(img, text, (x, y), 
                font, font_scale, (255, 255, 255), 2)

def hex_to_hsv(hex_color):
    """Konversi warna HEX ke HSV"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    bgr = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return hsv[0][0]

# Membaca dataset
df = pd.read_csv('colornames.csv')

# Inisialisasi dictionary untuk menyimpan range warna
color_ranges = {}

# Membuat range untuk setiap warna
for _, row in df.iterrows():
    hsv = hex_to_hsv(row['hex'])
    color_ranges[row['color_name']] = create_color_range(hsv, row['color_name'])

# Memulai capture video
cap = cv2.VideoCapture(0)

# Membuat window untuk trackbar
cv2.namedWindow('Settings')
cv2.createTrackbar('Min Area', 'Settings', 500, 5000, lambda x: None)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error membaca frame")
        break
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    min_area = cv2.getTrackbarPos('Min Area', 'Settings')
    
    # Untuk setiap warna dalam dataset
    for color_name, ranges in color_ranges.items():
        # Handle kasus khusus untuk coklat
        if color_name in ['Coklat', 'MerahMarun']:
            mask1 = cv2.inRange(hsv_frame, ranges['lower1'], ranges['upper1'])
            mask2 = cv2.inRange(hsv_frame, ranges['lower2'], ranges['upper2'])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv_frame, ranges['lower'], ranges['upper'])
        
        # Menghilangkan noise
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Mencari contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Tracking objek
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Menggambar kotak
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Menampilkan nama warna dengan background
                text = f" {color_name} "  # Menambah spasi di awal dan akhir teks
                put_text_with_background(frame, text, (x, y - 10))
                
                # Menampilkan titik tengah
                center_x = x + w//2
                center_y = y + h//2
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
    
    # Menampilkan frame
    cv2.imshow("Color Tracking", frame)
    
    # Keluar jika 'q' ditekan
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Membersihkan
cap.release()
cv2.destroyAllWindows()