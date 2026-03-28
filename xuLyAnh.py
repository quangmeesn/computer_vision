import cv2 as cv
import numpy as np
import urllib.request

# def load_image_from_url(url):
#     req = urllib.request.urlopen(url)
#     img_rw = np.asarray(bytearray(req.read()), dtype=np.uint8)
#     img = cv.imdecode(img_rw, 1)
#     return img

# def add_noise(img):
#     mean = 0
#     sigma = 50
#     noisy = np.random.normal(mean, sigma, img.shape)
#     new_img = np.clip(img + noisy, 0, 255).astype(np.uint8)
#     return new_img

# def add_muoi_tieu(img, ratio = 0.02):
#     nosy = img.copy()
#     soLuong = int(ratio*img.size)
#     #cho muoi vao 
#     toado = [np.random.randint(0, img.shape[0]-1, soLuong) for i in img.shape]
#     nosy [toado[0], toado[1], :] = 255
#     #cho tieu vao
#     toado = [np.random.randint(0, img.shape[0]-1, soLuong) for i in img.shape]
#     nosy [toado[0], toado[1], :] = 0

#     return nosy

# if __name__ == "__main__":
#     url = "https://raw.githubusercontent.com/opencv/opencv/refs/heads/4.x/samples/data/lena.jpg"
#     img = load_image_from_url(url)
#     cv.imshow('Loaded Image', img)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

    # img2 = add_noise(img)
    # cv.imshow("img2", img2)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    
    # img3 = np.concatenate((img, img2), axis=1)
    # cv.imshow("img3", img3)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # img_mieu_tieu = add_muoi_tieu(img, ratio=0.03)
    # cv.imshow("img_mieu_tieu", img_mieu_tieu)
    # cv.waitKey(0)       
    # cv.destroyAllWindows()
    
    # img4 = np.concatenate((img, img_mieu_tieu), axis=1)
    # cv.imshow("img4", img4)     
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # --------------------------------------------------------------------


def read_img_url(url):
    req = urllib.request.urlopen(url)
    img_rw = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv.imdecode(img_rw, 1)
    return img

def add_muoi_tieu(img, ratio = 0.02):
    nosy = img.copy()
    soLuong = int(ratio*img.size)
    #cho muoi vao 
    toado = [np.random.randint(0, i-1, soLuong) for i in img.shape]
    nosy [toado[0], toado[1], :] = 255
    #cho tieu vao
    toado = [np.random.randint(0, i-1, soLuong) for i in img.shape]
    nosy [toado[0], toado[1], :] = 0

    return nosy

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/udacity/CarND-LaneLines-P1/master/test_images/solidWhiteCurve.jpg"
    img = read_img_url(url)
    img2 = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    cv.imshow('Loaded Image', img2)
    cv.waitKey(0)
    cv.destroyAllWindows()


    # img_mieu_tieu = add_muoi_tieu(img, ratio=0.03)
    # cv.imshow("img_mieu_tieu", img_mieu_tieu)
    # cv.waitKey(0)       
    # cv.destroyAllWindows()

    # img1 =img_mieu_tieu.copy()
    # clean_img = cv.medianBlur(img1, 5)
    # cv.imshow("clean_img", clean_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # ed1 = cv.Canny(img_mieu_tieu, 20, 150)
    # ed2 = cv.Canny(clean_img, 20, 150)
    # ed3 = cv.Canny(img, 20, 150)
    # # 1. Lấy kích thước ảnh gốc
    # h, w = ed1.shape[:2]
    # # 2. Tính toán kích thước mới (ví dụ: giảm xuống còn 1/3)
    # new_size = (int(w/3), int(h/3))

    # # 3. Resize cả 3 ảnh
    # ed1_res = cv.resize(ed1, new_size)
    # ed2_res = cv.resize(ed2, new_size)
    # ed3_res = cv.resize(ed3, new_size)
    # img7 = np.concatenate((ed1_res, ed2_res, ed3_res), axis=1)
    # cv.imshow("Canny Edges: noisy | cleaned | original", img7)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    edge = cv.Canny(img2, 50, 150)
    cv.imshow("edge", edge)
    cv.waitKey(0)
    cv.destroyAllWindows()

    h,w = edge.shape
    mask = np.zeros_like(edge)
    polygon = np.array([[ (0,h), (w,h), (w//2 - 50, h//2), (w//2 + 50, h//2) ]], dtype=np.int32)
    cv.fillPoly(mask, polygon, 255)
    roi = cv.bitwise_and(edge, mask)
    cv.imshow("ROI",roi)
    cv.waitKey(0)
    cv.destroyAllWindows()

    lines = cv.HoughLinesP(
        roi,
        rho=1.0,
        theta=np.pi/180,
        threshold=50,
        minLineLength=50,
        maxLineGap=100
    )
    
    img_line = img.copy()
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv.line(img_line, (x1,y1), (x2,y2), (0,0,255),1)
            
    cv.imshow("detected_lane", img_line)
    cv.waitKey()
    cv.destroyAllWindows()


