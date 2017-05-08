import numpy as np
import cv2

from .sobel import my_sobel


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

class ColorThreshold:
    def __init__(self):
        raise Exception('abstract class')

    @classmethod
    def is_yellow(cls, img):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        low_yellow = np.array([0, 0, 0])
        high_yellow = np.array([0, 0, 0])
        mask = cv2.inRange(lab, low_yellow, high_yellow)
        return mask

    @classmethod
    def is_white(cls, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        low_white = np.array([0, 0, 0])
        high_white = np.array([0, 0, 0])
        mask = cv2.inRange(hsv, low_white, high_white)
        return mask

    @classmethod
    def do_thresholding(cls, image):
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        sunny_pixels, bright_pixels, mean = cls._detect_brightness(hls[:, :, 2])

        min_brightness = 40
        if sunny_pixels > 15000 or bright_pixels > 15000: # sunny / very bright
            return cls.__sunny_threshold(image, min_brightness)
        elif bright_pixels > 4000: # bright
            return cls.__bright_threshold(image, 2*min_brightness)
        else: # dark / shadow
            return cls.__normal_threshold(image, min_brightness)

    @staticmethod
    def _detect_brightness(image):
        height = image.shape[0]
        bottom = image[height//2:height, :]
        sun = np.zeros_like(bottom)
        sun[bottom > 240] = 1
        bright = np.zeros_like(bottom)
        bright[bottom > 200] = 1
        return sun.sum(), bright.sum(), bottom.mean()

    @staticmethod
    def _threshold(image, channel, min_thresh, max_thresh):
        result = np.zeros(image.shape[:2])
        result[(image[..., channel] >= min_thresh) & (image[..., channel] <= max_thresh)] = 1
        return result

    @classmethod
    def __sunny_threshold(cls, image, min_brightness=40):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        binary_lab_b = cls._threshold(lab, 2, 150, 255)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        binary_hsv_s = cls._threshold(hsv, 1, 110, 255)
        sobel = my_sobel(lab[..., 0])
        binary_sobel = np.zeros_like(sobel)
        binary_sobel[sobel > 80] = 1

        result = np.max((binary_lab_b, binary_hsv_s, binary_sobel), axis=0)
        result[lab[..., 0] < 2*min_brightness] = 0
        return result

    @classmethod
    def __bright_threshold(cls, image, min_brightness=40):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        binary_lab_b = cls._threshold(lab, 2, 150, 255)
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        binary_hls_s = cls._threshold(hls, 2, 90, 255)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        binary_hsv_s = cls._threshold(hsv, 1, 110, 255)
        binary_rgb_r = cls._threshold(image, 0, 220, 255)

        result = np.max((binary_lab_b, binary_hls_s, binary_hsv_s, binary_rgb_r), axis=0)
        result[lab[..., 0] < min_brightness] = 0
        result[(lab[..., 0] > 20) & (lab[..., 0] < 50) & (lab[..., 2] > 120)] = 1
        return result

    @classmethod
    def __normal_threshold(cls, image, min_brightness=40):
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        binary_lab_b = cls._threshold(lab, 2, 150, 255)
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        binary_hls_s = cls._threshold(hls, 2, 90, 255)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        binary_hsv_s = cls._threshold(hsv, 1, 110, 255)
        binary_rgb_r = cls._threshold(image, 0, 200, 255) # 170 or 200?

        result = np.max((binary_lab_b, binary_hls_s, binary_hsv_s, binary_rgb_r), axis=0)
        result[lab[..., 0] < min_brightness] = 0
        return result
