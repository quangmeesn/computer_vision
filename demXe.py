import cv2 as cv
from ultralytics import YOLO

# --- CẤU HÌNH DỄ CHỈNH SỬA ---
VIDEO_SOURCE = "mvideo.mp4"
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
LINE_COLOR = (0, 0, 255)        # Màu đỏ cho vạch
DOT_COLOR = (0, 255, 255)       # Màu vàng cho dấu chấm tâm
TEXT_COLOR = (255, 255, 255)     # Màu trắng cho ID
# ----------------------------

model = YOLO("yolov8n.pt")
cap = cv.VideoCapture(VIDEO_SOURCE)

count = 0
counted_ids = set()
prev_y = {}
# frame_count = 0  
roi_x1, roi_y1 = 0, 0
roi_x2, roi_y2 = 0, 0
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret: break
    
    

    # Tự động đặt vạch ở 2/3 khung hình (dễ chỉnh sửa số 0.7)
    line_y = int(frame.shape[0] * 0.7)

    # Tracking tối ưu: chỉ lấy class xe, không hiện log (verbose=False)
    results = model.track(frame, persist=True, classes=VEHICLE_CLASSES, verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            # area = (x2 - x1) * (y2 - y1)
            # if area < 2000:
            #      continue
            # cv.circle(frame, (x1, y1), 4, (0, 255, 255), -1)
            # cv.circle(frame, (x2, y1), 4, (0, 255, 255), -1)
            # cv.circle(frame, (x1, y2), 4, (0, 255, 255), -1)
            # cv.circle(frame, (x2, y2), 4, (0, 255, 255), -1)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # 1. Vẽ Box, Dấu chấm tâm và ID
            # color = (0, 255, 0) if track_id % 2 == 0 else (255, 0, 0)
            # cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv.circle(frame, (cx, cy), 5, DOT_COLOR, -1)
            
            
            # Vẽ ID phía trên box, màu trắng, font nhỏ gọn
            cv.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
            # tăng kích thước chỉnh 0.6 và 2
            # ------------------------------

            # 2. Logic đếm (So sánh vị trí Y frame trước và hiện tại qua vạch)
            if track_id in prev_y:
                if (prev_y[track_id] < line_y <= cy) or (prev_y[track_id] > line_y >= cy):
                    if track_id not in counted_ids:
                        count += 1
                        counted_ids.add(track_id)
            
            prev_y[track_id] = cy

    # 3. Hiển thị UI
    cv.line(frame, (0, line_y), (frame.shape[1], line_y), LINE_COLOR, 3)
    cv.putText(frame, f"VEHICLES: {count}", (20, 50), 
               cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv.imshow("YOLOv8 Counter", frame)
    if cv.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv.destroyAllWindows()