import numpy as np
from cv2 import cv2

def blot(face):
    face = np.zeros(face.shape, dtype = np.uint8)
    return face

def blur(face):
    face = cv2.GaussianBlur(face, (31, 31), 6)
    return face