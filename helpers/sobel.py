import numpy as np
import cv2

from .calibration import rgb2gray

def grayscale(image):
    if len(image.shape) == 3:
        return rgb2gray(image)
    return image

def sobel(axis, image, axis, directional=False, threshold=(20, 100), kernel=3):
    gray = grayscale(image)
    s = None
    if not directional:
        if axis == 'x':
            s = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
        elif axis == 'y':
            s = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)
        elif axis == 'xy':
            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)
            s = np.sqrt(sx**2 + sy**2)
        else:
            raise 'Invalid value "{}" for axis'.format(axis)
        s = np.absolute(s)
        s = np.uint8(255 * s / np.max(s))
    else:
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)
        s = np.arctan2(np.absolute(sy), np.absolute(sx))

    binary = np.zeros_like(gray)
    binary[(s >= threshold[0]) & (s <= threshold[1])] = 1
    return binary

