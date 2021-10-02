import numpy as np
from cv2 import cv2

width = 640
height = 480
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

def blot(face):
    face = np.zeros(face.shape, dtype = np.uint8)
    return face

def blur(face):
    face = cv2.GaussianBlur(face, (31, 31), 6)
    return face

def preprocessing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    gray = gray.astype(np.uint8)
    faceRects = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    if type(faceRects)!=tuple:
        for (fX, fY, fW, fH) in faceRects:
            #faceROI = gray[fY:fY+ fH, fX:fX + fW]
            face = img[fY:fY+ fH, fX:fX + fW, :]
            img[fY:fY+ fH, fX:fX + fW, :] = blur(face)
    return img

while True:
    success, originalimg = cap.read()
    img = preprocessing(originalimg)
    cv2.imshow('output', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break