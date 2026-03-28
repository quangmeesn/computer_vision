

import cv2 as cv
import numpy as np
import smtplib
from email.message import EmailMessage
import ssl
import time

# =========================
# CẤU HÌNH EMAIL
# =========================
sender = "quangmeesn2002@gmail.com"
password = "dehp qdjl cese lnyd"
receiver = "daoquangmen73@gmail.com"

def send_email(person_name):

    msg = EmailMessage()
    msg["Subject"] = "Canh bao he thong nhan dien"
    msg["From"] = sender
    msg["To"] = receiver
    msg.set_content(f"He thong phat hien: {person_name}")

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com",465,context=context) as server:
        server.login(sender,password)
        server.send_message(msg)

    print("Da gui email!")

# =========================
# LIVENESS DETECTION
# =========================
prev_face = None

def is_real_face(face_img):

    global prev_face

    face_img = cv.resize(face_img,(200,200))

    if prev_face is None:
        prev_face = face_img.copy()
        return False

    diff = cv.absdiff(prev_face, face_img)

    score = np.sum(diff)

    prev_face = face_img.copy()

    if score > 40000:
        return True
    else:
        return False

# =========================
# LOAD MODEL
# =========================
recog_tool = cv.face.LBPHFaceRecognizer_create()
recog_tool.read("face_recognizer_model.yml")

label_dict = np.load("label_dict.npy", allow_pickle=True).item()

face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# CAMERA
# =========================
cap = cv.VideoCapture(0)

last_email_time = 0
email_delay = 15

while True:

    ret, frame = cap.read()

    if not ret:
        print("Loi camera")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    blur = cv.GaussianBlur(gray,(5,5),0)

    faces = face_cascade.detectMultiScale(blur,1.3,5)

    for (x,y,w,h) in faces:

        face_img = blur[y:y+h, x:x+w]

        face_img = cv.resize(face_img,(200,200))

        # chống ảnh tĩnh
        if not is_real_face(face_img):

            cv.putText(frame,
                       "No Motion",
                       (x,y-10),
                       cv.FONT_HERSHEY_SIMPLEX,
                       0.7,
                       (255,0,0),
                       2)

            continue

        label, confidence = recog_tool.predict(face_img)

        if confidence < 50:

            name = label_dict[label]
            color = (0,255,0)

        else:

            name = "Unknown"
            color = (0,0,255)

            if time.time() - last_email_time > email_delay:

                send_email("Nguoi la")
                last_email_time = time.time()

        cv.rectangle(frame,(x,y),(x+w,y+h),color,2)

        cv.putText(frame,
                   f"{name} {int(confidence)}",
                   (x,y-10),
                   cv.FONT_HERSHEY_SIMPLEX,
                   0.8,
                   color,
                   2)

    cv.imshow("Face Recognition System",frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv.destroyAllWindows()