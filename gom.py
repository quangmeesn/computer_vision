import os
import glob

# 1. Điền đường dẫn tới thư mục chứa file .txt của dataset Gà vào đây
# (Nhớ dùng dấu '/' thay vì '\' trên Windows để tránh lỗi đường dẫn)
labels_dir = 'C:/Users/ASUS/Desktop/GOD,CHICKEN/Chicken.v1i.yolov8/valid/labels' 

# Tìm toàn bộ file .txt trong thư mục
txt_files = glob.glob(os.path.join(labels_dir, '*.txt'))

count = 0
for file_path in txt_files:
    # Đọc nội dung file hiện tại
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split() # Tách dòng thành một list các chữ số
        
        # Kiểm tra xem dòng có đủ 5 thành phần chuẩn YOLO không
        if len(parts) >= 5: 
            if parts[0] == '0':
                parts[0] = '1' # Chỉ đổi phần tử đầu tiên (Class ID) từ 0 sang 1
            
            # Ghép lại thành chuỗi và thêm dấu xuống dòng
            new_line = ' '.join(parts) + '\n'
            new_lines.append(new_line)
        else:
            new_lines.append(line) # Giữ nguyên nếu là dòng trống
            
    # Ghi đè nội dung mới lại vào chính file đó
    with open(file_path, 'w') as file:
        file.writelines(new_lines)
    
    count += 1

print(f"Xong! Đã đổi thành công Class ID cho {count} file.")