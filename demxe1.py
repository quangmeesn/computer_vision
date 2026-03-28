import cv2 as cv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv.VideoCapture("mvideo.mp4")
unique_vehicles = set()
# hiển thị số frame đã xử lý
# đổi màu chữ count sang xanh
# tăng kich thước chữ label 
# vẽ đường ngang ở giữa màn hình 
# ẩn text id
# chỉ đếm bbox > 2000
# chỉ đếm xe vùng bên trái
# chỉ đếm xe vùng bên phải
# chỉ đếm một loại xe nhất định
# đổi màu bbox theo id chẵn/ lẻ
# hiển thị số xe hiện tại trên ROI 
# Hiển thị ID lớn nhất
# vẽ đường ROI để đếm xe
# vẽ chấm tròn ở bốn góc của bbox
# hiển thị traffic jam nếu số xe  > 10
frame_count = 0
while True: 
    ret, frame = cap.read()
    if not ret: 
        break
    frame_count += 1
    h, w, _ = frame.shape
    results = model.track(frame, persist=True)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy
        ids = results[0].boxes.id
        classes = results[0].boxes.cls

        for box, track_id, cls in zip(boxes, ids, classes):
            x1, y1, x2, y2 = map(int, box)
            track_id = int(track_id)

            label = model.names[int(cls)]

            if label not in ["car", "motorcycle", "bus", "truck"]:
                continue
            

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

        

            cv.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            unique_vehicles.add(track_id)
            # Vẽ bbox

            color = (0, 255, 0) 
            cv.put
            cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv.putText(frame, f"{label}ID:{track_id}", 
                       (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # hiển thị 
            cv.putText(frame, f"Unique Vehicles: {len(unique_vehicles)}", 
                            (30, 20),
                            cv.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 255),
                            2)
    
    cv.imshow("Vehicle Tracking", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
