import cv2 as cv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv.VideoCapture("mvideo.mp4")

unique_vehicles = set()
frame_count = 0

VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w, _ = frame.shape

    # ===== VẼ ĐƯỜNG NGANG GIỮA =====
    mid_y = h // 2
    cv.line(frame, (0, mid_y), (w, mid_y), (255, 0, 0), 2)

    # ===== VẼ ROI (vùng giữa) =====
    roi_x1, roi_x2 = w // 4, 3 * w // 4
    roi_y1, roi_y2 = h // 4, 3 * h // 4
    cv.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)

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

            # ===== CHỈ ĐẾM 1 LOẠI XE (ví dụ: car) =====
            if label != "car":
                continue

            # ===== CHỈ LẤY XE =====
            if label not in VEHICLE_CLASSES:
                continue

            # ===== LỌC DIỆN TÍCH > 2000 =====
            area = (x2 - x1) * (y2 - y1)
            if area < 2000:
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # ===== CHỈ ĐẾM BÊN TRÁI =====
            if cx > w // 2:
                continue

            # (Nếu muốn bên phải thì đổi lại)
            # if cx < w // 2:
            #     continue

            unique_vehicles.add(track_id)

            # ===== ĐỔI MÀU BBOX THEO ID =====
            color = (0, 255, 0) if track_id % 2 == 0 else (0, 0, 255)

            # ===== VẼ BBOX =====
            cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # ===== TĂNG SIZE LABEL & ẨN ID =====
            cv.putText(frame, f"{label}",
                       (x1, y1 - 10),
                       cv.FONT_HERSHEY_SIMPLEX,
                       1.0,
                       (255, 255, 255),
                       2)

            # ===== VẼ 4 GÓC BBOX =====
            cv.circle(frame, (x1, y1), 4, (0, 255, 255), -1)
            cv.circle(frame, (x2, y1), 4, (0, 255, 255), -1)
            cv.circle(frame, (x1, y2), 4, (0, 255, 255), -1)
            cv.circle(frame, (x2, y2), 4, (0, 255, 255), -1)

            # ===== ĐẾM XE TRONG ROI =====
            if roi_x1 < cx < roi_x2 and roi_y1 < cy < roi_y2:
                current_in_roi += 1

            # ===== ID LỚN NHẤT =====
            if track_id > max_id:
                max_id = track_id

    # ===== HIỂN THỊ FRAME COUNT =====
    cv.putText(frame, f"Frame: {frame_count}",
               (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ===== ĐỔI MÀU COUNT XANH =====
    cv.putText(frame, f"Total Vehicles: {len(unique_vehicles)}",
               (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ===== XE TRONG ROI =====
    cv.putText(frame, f"In ROI: {current_in_roi}",
               (20, 90), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # ===== HIỂN THỊ MAX ID =====
    cv.putText(frame, f"Max ID: {max_id}",
               (20, 120), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # ===== TRAFFIC JAM =====
    if current_in_roi > 5:
        cv.putText(frame, "TRAFFIC JAM!",
                   (w // 3, h // 2),
                   cv.FONT_HERSHEY_SIMPLEX,
                   1.5,
                   (0, 0, 255),
                   3)

    cv.imshow("Vehicle Tracking", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()