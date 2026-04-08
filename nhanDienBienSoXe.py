
import cv2 as cv
import easyocr

img_path = 'car2.jpg'  # Đường dẫn đến ảnh xe của bạn

img = cv.imread(img_path, cv.IMREAD_COLOR)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

binary = cv.threshold(gray, 175, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

# khử nhiễu
clean_img = cv.fastNlMeansDenoising(binary, h=10)

edges = cv.Canny(clean_img, 30, 200)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
closed_bbox = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

contours, _ = cv.findContours(closed_bbox, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

plates = []
img_size = img.shape[0] * img.shape[1]

# OCR init (CHỈ 1 LẦN)


def crop_plate(x, y, w, h, pad=5):
    x1 = max(0, x-pad)
    y1 = max(0, y-pad)
    x2 = min(img.shape[1], x+w+pad)
    y2 = min(img.shape[0], y+h+pad)
    return img[y1:y2, x1:x2]

for cnt in contours:
    x, y, w, h = cv.boundingRect(cnt)
    area = w * h
    aspect_ratio = w / h
    area_ratio = area / img_size  # FIX

    if (2.0 < aspect_ratio < 5.5) and (0.005 < area_ratio < 0.2):
        plates.append((x,y,w,h))
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

# OCR từng plate
for i, (x, y, w, h) in enumerate(plates):
    plate_img = crop_plate(x, y, w, h)

    # preprocess tốt hơn cho OCR
    plate_gray = cv.cvtColor(plate_img, cv.COLOR_BGR2GRAY)
    plate_gray = cv.resize(plate_gray, None, fx=2, fy=2)

    _, plate_bin = cv.threshold(
        plate_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
    )
    reader = easyocr.Reader(['en'], gpu=False, download_enabled=False) 
    result = reader.readtext(
        plate_bin,
        allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        detail=1
    )

    for (bbox, text, conf) in result:
        print(f"[Plate {i}] Text: {text} | Confidence: {conf:.2f}")

    cv.imshow(f'plate_{i}', plate_bin)
    cv.waitKey(0)

cv.imshow('result', img)
cv.waitKey(0)
cv.destroyAllWindows()
