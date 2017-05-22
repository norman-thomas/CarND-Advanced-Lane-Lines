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
    def threshold(cls, img):
        #return cls._simple_threshold(img)
        return cls._do_thresholding(img)

    @classmethod
    def _other_threshold(cls, img):
        yellow = cls.is_yellow(img)
        white = cls.is_white(img)
        return np.max((yellow, white), axis=0)

    @classmethod
    def is_yellow(cls, img):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        mask = cv2.inRange(lab, (20, 100, 100), (50, 255, 255))
        return mask

    @classmethod
    def is_white(cls, img):
        sensitivity_1 = 68
        sensitivity_2 = 60

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        white_1 = cv2.inRange(hsv, (0, 0, 255 - sensitivity_1), (255, 20, 255))
        white_2 = cv2.inRange(hls, (0, 255 - sensitivity_2, 0), (255, 255, sensitivity_2))
        white_3 = cv2.inRange(img, (200, 200, 200), (255, 255, 255))

        mask = np.max((white_1, white_2, white_3), axis=0)
        return mask

    @classmethod
    def _simple_threshold(cls, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

        thresh_lab = cv2.inRange(lab, (100, 0, 145), (255, 255, 255))
        thresh_hls = cv2.inRange(hls, (0, 40, 165), (100, 255, 255))
        thresh_hsv = cv2.inRange(hsv, (0, 0, 235), (30, 255, 255))
        
        mask = np.max((thresh_lab, thresh_hls, thresh_hsv), axis=0)
        return mask


    @classmethod
    def _do_thresholding(cls, image):
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
