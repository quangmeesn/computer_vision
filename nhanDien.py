from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    # 1. Khởi tạo model (Dùng YOLO26 cho mới nhất hoặc YOLOv8)
    model = YOLO("yolo8s.pt") 

    # 2. Train model
    model.train(
        data="C:/Users/ASUS/Desktop/computer_vision/chicken.v1i.yolov11/data.yaml",     # Đường dẫn file yaml
        epochs=100,           # Số vòng lặp
        imgsz=640,            # Kích thước ảnh
        batch=8,             # Số lượng ảnh mỗi đợt (giảm xuống 4 hoặc 8 nếu máy yếu)
        device=1,             # Chuyển thành device='cpu' nếu không có card NVIDIA
        workers=2             # Số luồng xử lý dữ liệu
    )