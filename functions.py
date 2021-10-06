import numpy as np
from cv2 import cv2

def blot(face):
    face = np.zeros(face.shape, dtype = np.uint8)
    return face

def blur(face, level):
    face = cv2.GaussianBlur(face, (31+((level-1)*10), 31+((level-1)*10)), 6)
    return face

def clown(face, you):
    you = cv2.resize(you, (face.shape[0], face.shape[1]))
    for i in range(face.shape[0]):
        for j in range(face.shape[1]):
            if float(you[i][j][0])+float(you[i][j][1])+float(you[i][j][2])!=0:
                face[i][j]=you[i][j]
    return face