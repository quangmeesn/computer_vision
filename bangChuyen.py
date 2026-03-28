# import cv2 as cv
# import numpy as np


# vid = cv.VideoCapture("bang_chuyen.mp4")
# # base_frame = None
# # while True:
# #     ret, frame = vid.read()
# #     if not ret:
# #         break
# #     if frame is not None:
        
# #         cv.imshow("video", frame)
# #     if cv.waitKey(100) == ord("q"):
# #         break

# base_frame = None
# while True:
#     ret, frame = vid.read()
#     if not ret:
#         break
#     if frame is not None:
#         gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#         gray = cv.GaussianBlur(gray, (21, 21), 0)

#     if base_frame is None:
#         base_frame = gray
#         continue
#     # tinfh toan su khac biet giua frame hien tai va frame co san
#     chenh_lech = cv.absdiff(base_frame, gray)
#     nguong = cv.threshold(chenh_lech, 50, 255, cv.THRESH_BINARY)[1]
#     nguong = cv.dilate(nguong, None, iterations=2)
#     bien, info  = cv.findContours(nguong.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#     for b in bien:
#         if cv.contourArea(b) < 800:
#             continue
#         x, y, w, h = cv.boundingRect(b)
        
#         cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv.imshow("Webcam", frame)
#         cv.imshow("Chenh Lech", nguong)
#     if cv.waitKey(100) & 0xFF == ord('q'):
#         break  
      
     
#  #đếm số lượng hình tròn vượt qua line màu đỏ.   
# cv.destroyAllWindows()

# --------------------------------------------------------------
# import cv2 as cv
# import numpy as np

# cap = cv.VideoCapture("bang_chuyen.mp4") #thay tên file = kênh camera 0, ip,...
# count = 0 # biến đếm
# vat_the = []   # danh sách vật thể
# next_id = 0
# line_x = 600
# DIST_THRESHOLD = 50

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY) #chuyển từ màu sang đen trắng 
#     gray = cv.medianBlur(gray, 5) #làm sạch nhiễu

#     circles = cv.HoughCircles(
#         gray,
#         cv.HOUGH_GRADIENT,
#         dp =1,
#         minDist = 20,
#         param1= 50,
#         param2= 30,
#         minRadius= 5,
#         maxRadius= 50
#     )

#     if circles is not None:
#         circles = np.round(circles).astype(int)

#         for circle in circles[0, :]:
#             x, y, r = circle
#             cv.circle(frame, (x, y), r, (0, 0, 255), 2)

#             matched = False

#             # so khớp với vật thể cũ
#             for obj in vat_the:
#                 if abs(obj["x"] - x) < DIST_THRESHOLD:
#                     obj["x"] = x
#                     matched = True

#                     # ĐẾM ĐÚNG 1 LẦN
#                     if not obj["counted"] and x > line_x:
#                         count += 1
#                         obj["counted"] = True
#                         print(f"Vat the thu {count} da di qua")

#                     break

#             # nếu là vật thể mới
#             if not matched:
#                 vat_the.append({
#                     "id": next_id,
#                     "x": x,
#                     "counted": x > line_x
#                 })
#                 next_id += 1

#     cv.imshow("f", frame)
#     if cv.waitKey(10) == ord('q'):
#         break
# cv.destroyAllWindows()



# -------------------------------------------------------------
import cv2 as cv
import numpy as np

cap = cv.VideoCapture("bang_chuyen.mp4")

LINE_X = 600
count = 0

objects = {}
obj_id = 0
counted = set()

def dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (9, 9), 1.5)

    circles = cv.HoughCircles(
        gray,
        cv.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=100,
        param2=20,
        minRadius=8,
        maxRadius=60
    )

    detections = []

    if circles is not None:
        circles = np.around(circles[0]).astype(int)

        for x, y, r in circles:
            detections.append((x, y))

            # vẽ hình vuông quanh hình tròn
            cv.rectangle(
                frame,
                (x - r, y - r),
                (x + r, y + r),
                (0, 255, 0),
                2
            )

    new_objects = {}

    for x, y in detections:
        matched = False
        for oid, (px, py) in objects.items():
            if dist((x, y), (px, py)) < 30:
                new_objects[oid] = (x, y)
                matched = True

                if px < LINE_X and x >= LINE_X and oid not in counted:
                    count += 1
                    counted.add(oid)
                    print(f"Phát hiện vật thể thứ {count}")
                break

        if not matched:
            new_objects[obj_id] = (x, y)
            obj_id += 1

    objects = new_objects

    # vẽ line đỏ
    cv.line(frame, (LINE_X, 0), (LINE_X, frame.shape[0]), (0, 0, 255), 2)
    cv.putText(frame, f"COUNT: {count}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow("Video", frame)

    if cv.waitKey(60) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()