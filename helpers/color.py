import numpy as np
import cv2


def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def split_channels(images):
    result = []
    for img in images:
        for channel in cv2.split(img):
            result.append(channel)
    return result

def binary(images, channel, threshold=(150,255)):
    def _binary(image):
        c = cv2.split(image)[channel]
        b = np.zeros_like(c)
        b[(c >= threshold[0]) & (c <= threshold[1])] = 1
        return b
    return np.array([_binary(img) for img in images])

def combine_thresholds(*binaries):
    if len(binaries) == 0:
        return None
    result = np.zeros_like(binaries[0])
    for binary in binaries:
        result[binary == 1] = 1
    return result

def find_maximum(arr):
    start = np.argmax(arr)
    end = start
    while end < len(arr) and arr[end] >= arr[start]:
        end += 1
    end -= 1
    return (start + end) // 2

def histogram(img, from_=None, to=None):
    height, width = img.shape
    if to is None:
        to = height
    if from_ is None:
        from_ = 0
    if from_ >= to or to > height:
        return None
    area = img[from_:to, :]
    return np.sum(area, axis=0)

