from ultralytics import YOLO
import cv2

# 1. Load model đã train của bạn
model = YOLO("C:\\Users\\ASUS\\Desktop\\computer_vision\\23150502.pt") 

# 2. Mở file video
video_path = "C:\\Users\\ASUS\\Desktop\\computer_vision\\dogs.mp4" # Thay bằng đường dẫn video của bạn
cap = cv2.VideoCapture(video_path)

# Kiểm tra nếu không mở được video
if not cap.isOpened():
    print("Không thể mở video. Hãy kiểm tra lại đường dẫn!")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        # 3. Chạy YOLO trên từng khung hình (frame)
        # stream=True giúp xử lý mượt hơn cho video dài
        results = model(frame, stream=True)
        
        for r in results:
            # Vẽ các khung hình và nhãn lên frame gốc
            annotated_frame = r.plot()
            
            # 4. Hiển thị cửa sổ video trực tiếp
            cv2.imshow("YOLO Live Detection", annotated_frame)
        
        # Nhấn phím 'q' trên bàn phím để thoát cửa sổ
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Giải phóng bộ nhớ
cap.release()
cv2.destroyAllWindows()