import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import time
import math
# vẽ một đường thẳng từ góc trên bên trái đến góc dưới bên phải của ảnh,
#  sử dụng màu trắng và độ dày 1 pixel.
# M = 1024
# N = 768

# img = np.zeros((N,M), dtype=np.uint8)
# cv.line(img, (0,0), (M-1, N-1), 255)
# cv.imshow('image', img )
# if cv.waitKey(0) == 27:
#     cv.destroyAllWindows()

#Sử dụng CV vẽ một đường tròn trùng tâm với tâm của ảnh, có bán kính là 100, màu trắng, độ dày 2 pixel.
#
# m = 1024
# n = 768
# c =3
# cl_img = np.zeros((n,m,c), dtype=np.uint8)
# cl_img[10:300, :, 0] = 255
# cl_img[250:500, :, 1] = 255
# cl_img[250:700, :, 2] = 255
# cv.imshow('color image', cl_img)
# cv.waitKey(0)
# cv.destroyAllWindows()

#vẽ một bàn cờ vua có kích thước 8x8, mỗi ô có kích thước 100x100 pixel. Sử dụng hai màu đen và trắng để tô màu các ô.
# chess_board = np.zeros((800,800,3), dtype=np.uint8)
# for i in range(8):
#     for j in range(8):  
#         if (i+j) % 2 == 0:
#             chess_board[i*100:(i+1)*100, j*100:(j+1)*100] = 255
# cv.imshow('chess board', chess_board)
# cv.waitKey(0)
# cv.destroyAllWindows()


#Vẽ mặt đồng hồ hình tròn, nền màu tím, có các số dạng la mã màu sắc khác nhau. 
#Có 3 kim đồng hồ: Giờ, phút, giây.
#Kim giờ màu xanh dương, kim phút màu xanh lá cây, kim giây màu đỏ.
#level 2: Vẽ kim giây chuyển động.
#level 3: Vẽ kim phút chuyển động.
#level 4: Vẽ kim giờ chuyển động.   
#level 5: Vẽ thêm các vạch chỉ phút trên mặt đồng hồ. Và kim giây, kim giờ, kim phút, hoạt động theo logic

clock_face = np.zeros((600,600,3), dtype=np.uint8)
clock_face[:] = (255, 0, 255)  # Màu tím nền
cv.circle(clock_face, (300, 300), 250, (255, 255, 255), 5)  # Viền đồng hồ
font = cv.FONT_HERSHEY_SIMPLEX
# Vẽ số la mã
roman_numerals = ['XII', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI']
for i, numeral in enumerate(roman_numerals):
    angle = i * 30  # 360/12
    x = int(300 + 200 * np.sin(np.radians(angle)))
    y = int(300 - 200 * np.cos(np.radians(angle)))
    cv.putText(clock_face, numeral, (x-15, y+10), font, 0.7, (0, 255, 255), 2)
# Vẽ kim đồng hồ
def draw_hand(image, angle, length, color, thickness):
    x = int(300 + length * np.sin(np.radians(angle)))
    y = int(300 - length * np.cos(np.radians(angle)))
    cv.line(image, (300, 300), (x, y), color, thickness)
draw_hand(clock_face, 90, 100, (255, 0, 0), 8)  # Kim giờ
draw_hand(clock_face, 180, 150, (0, 255, 0), 6)  # Kim phút
draw_hand(clock_face, 270, 200, (0, 0, 255), 2)  # Kim giây
cv.imshow('clock face', clock_face) 
cv.waitKey(0)
cv.destroyAllWindows()


# ====== HẰNG SỐ ======
WIDTH, HEIGHT = 600, 600
CENTER = (300, 300)
RADIUS = 250

# ====== VẼ KIM ======
def draw_hand(image, angle, length, color, thickness):
    x = int(CENTER[0] + length * math.sin(math.radians(angle)))
    y = int(CENTER[1] - length * math.cos(math.radians(angle)))
    cv.line(image, CENTER, (x, y), color, thickness)

# ====== VẼ VẠCH PHÚT ======
def draw_minute_marks(image):
    for i in range(60):
        angle = i * 6
        outer_x = int(CENTER[0] + RADIUS * math.sin(math.radians(angle)))
        outer_y = int(CENTER[1] - RADIUS * math.cos(math.radians(angle)))

        if i % 5 == 0:   # vạch giờ
            inner_len = 30
            thickness = 4
        else:            # vạch phút
            inner_len = 15
            thickness = 2

        inner_x = int(CENTER[0] + (RADIUS - inner_len) * math.sin(math.radians(angle)))
        inner_y = int(CENTER[1] - (RADIUS - inner_len) * math.cos(math.radians(angle)))

        cv.line(image, (inner_x, inner_y), (outer_x, outer_y), (255,255,255), thickness)

# ====== VẼ SỐ LA MÃ ======
def draw_roman_numbers(image):
    roman_numerals = ['XII','I','II','III','IV','V','VI','VII','VIII','IX','X','XI']
    colors = [
        (0,0,255),(0,255,255),(0,255,0),(255,255,0),
        (255,0,0),(255,0,255),(255,255,255),(200,200,0),
        (0,200,200),(200,0,200),(180,180,255),(0,180,255)
    ]

    font = cv.FONT_HERSHEY_SIMPLEX

    for i, numeral in enumerate(roman_numerals):
        angle = i * 30
        x = int(CENTER[0] + 200 * math.sin(math.radians(angle)))
        y = int(CENTER[1] - 200 * math.cos(math.radians(angle)))
        cv.putText(image, numeral, (x-20, y+10), font, 0.8, colors[i], 2)

# ====== MAIN LOOP ======
while True:
    clock_face = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    clock_face[:] = (255, 0, 255)  # nền tím

    # Viền đồng hồ
    cv.circle(clock_face, CENTER, RADIUS, (255,255,255), 5)

    draw_minute_marks(clock_face)
    draw_roman_numbers(clock_face)

    # ====== LẤY THỜI GIAN ======
    t = time.localtime()
    sec = t.tm_sec
    minute = t.tm_min
    hour = t.tm_hour % 12

    # ====== GÓC KIM ======
    sec_angle = sec * 6
    min_angle = minute * 6 + sec * 0.1
    hour_angle = hour * 30 + minute * 0.5

    # ====== VẼ KIM ======
    draw_hand(clock_face, hour_angle, 100, (255,0,0), 8)    # giờ – xanh dương
    draw_hand(clock_face, min_angle, 150, (0,255,0), 6)     # phút – xanh lá
    draw_hand(clock_face, sec_angle, 200, (0,0,255), 2)     # giây – đỏ

    # Tâm đồng hồ
    cv.circle(clock_face, CENTER, 8, (0,0,0), -1)

    cv.imshow("Analog Clock - Level 5", clock_face)

    if cv.waitKey(1000) & 0xFF == 27:  # ESC để thoát
        break

cv.destroyAllWindows()