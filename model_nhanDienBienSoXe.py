import cv2 as cv
import easyocr
import time

# 1. Khởi tạo Reader một lần duy nhất bên ngoài
# Nếu có GPU NVIDIA, hãy để gpu=True
reader = easyocr.Reader(['en'], gpu=False)

video_path = 'baigiuxe.mp4' 
cap = cv.VideoCapture(video_path)

# Cấu hình tần suất OCR (Ví dụ: 10 frames mới OCR 1 lần để tránh lag)
OCR_FRAME_INTERVAL = 10
frame_count = 0
last_results = []

def preprocess_for_ocr(plate_img):
    """Tiền xử lý vùng biển số để OCR chính xác hơn"""
    gray = cv.cvtColor(plate_img, cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return binary

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    display_frame = frame.copy()
    img_size = frame.shape[0] * frame.shape[1]

    # --- BƯỚC 1: TÌM VÙNG BIỂN SỐ (Xử lý trên mọi frame) ---
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Dùng GaussianBlur thay vì fastNlMeans để tăng tốc độ trên video
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blur, 100, 200)
    
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    current_plates = []
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        aspect_ratio = w / h
        area_ratio = (w * h) / img_size

        # Lọc vùng tiềm năng là biển số
        if (0.7 < aspect_ratio < 5.0) and (0.0001 < area_ratio < 0.02):
            current_plates.append((x, y, w, h))
            cv.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # --- BƯỚC 2: NHẬN DIỆN CHỮ (Chỉ thực hiện mỗi khoảng Interval) ---
    if frame_count % OCR_FRAME_INTERVAL == 0:
        last_results = [] # Xóa kết quả cũ
        for (x, y, w, h) in current_plates:
            # Crop vùng biển số có thêm lề (padding)
            pad = 5
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(frame.shape[1], x+w+pad), min(frame.shape[0], y+h+pad)
            plate_roi = frame[y1:y2, x1:x2]

            if plate_roi.size > 0:
                plate_bin = preprocess_for_ocr(plate_roi)
                
                # Thực hiện OCR
                ocr_res = reader.readtext(plate_bin, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', detail=0)
                if ocr_res:
                    text = "".join(ocr_res)
                    last_results.append(((x, y), text))

    # --- BƯỚC 3: HIỂN THỊ KẾT QUẢ ---
    for (pos, text) in last_results:
        cv.putText(display_frame, text, (pos[0], pos[1] - 10), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv.imshow('License Plate Detection', display_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()