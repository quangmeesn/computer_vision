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

# vẽ đường ROI 

# vẽ chấm tròn ở bốn góc của bbox
# hiển thị traffic jam nếu số xe  > 10

while True: 
    ret, frame = cap.read()
    if not ret: 
        break

    frame_count += 1
    h, w, _ = frame.shape

    mid_y = h // 2
    cv.line(frame, (0, mid_y), (w, mid_y), (255, 0, 0), 2)

    # roi_x1, roi_x2 = w // 4, 3 * w // 4
    # roi_y1, roi_y2 = h // 4, 3 * h // 4
    # cv.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)

    results = model.track(frame, persist=True)

    current_in_roi = 0
    max_id = -1

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

            if label != "car":
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # if cx > w // 2:
            #     continue

            # if roi_x1 < cx < roi_x2 and roi_y1 < cy < roi_y2:
            #     current_in_roi += 1

            # if track_id > max_id:
            #     max_id = track_id

            # cv.circle(frame, (x1, y1), 4, (0, 0, 255), -1)
            # cv.circle(frame, (x2, y1), 4, (0, 0, 255), -1)
            # cv.circle(frame, (x1, y2), 4, (0, 0, 255), -1)
            # cv.circle(frame, (x2, y2), 4, (0, 0, 255), -1)

            cv.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            


            unique_vehicles.add(track_id)
            # Vẽ bbox

            color = (0, 255, 0) if track_id % 2 == 0 else (0, 0, 255)
            
            
            cv.putText(frame, f"Frame: {frame_count}", 
                       (30, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv.putText(frame, f"{label}ID:{track_id}", 
                       (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # hiển thị 
            cv.putText(frame, f"Unique Vehicles: {len(unique_vehicles)}", 
                            (30, 20),
                            cv.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 0),
                            2)
    # cv.putText(frame, f"in ROI: {current_in_roi}", 
    #                    (30, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    # cv.putText(frame, f"Max ID: {max_id}", 
    #                    (30, 110), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    if current_in_roi > 5:
        cv.putText(frame, "TRAFFIC JAM!", 
                   (w // 2 - 100, h // 2), 
                   cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
    cv.imshow("Vehicle Tracking", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
