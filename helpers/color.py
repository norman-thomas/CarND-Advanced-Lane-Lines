import numpy as np
import cv2


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

