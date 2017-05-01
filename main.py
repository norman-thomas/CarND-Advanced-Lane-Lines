import os
#import glob

import numpy as np
import cv2

import random
import math

import matplotlib.image as mpimg

import helpers.calibration as calibration
import helpers.sobel as sobel
import helpers.color as color
import helpers.warp as warp
import helpers.line as line

def save_images(images, folder, filenames):
    if not os.path.exists(folder):
        os.mkdir(folder)
    for img, fname in zip(images, filenames):
        if len(img.shape) == 2:
            mpimg.imsave(os.path.join(folder, fname), img, format='jpg', cmap='gray')
        else:
            mpimg.imsave(os.path.join(folder, fname), img, format='jpg')

def calibrate_camera(folder='camera_cal'):
    cal_files = os.listdir(folder)
    cal_files = list(map(lambda f: os.path.join(folder, f), cal_files))
    cal_images = [ calibration.load_image(f) for f in cal_files ]
    print('There are {} calibration images present'.format(len(cal_files)))

    cal_chessboards = [calibration.chessboard(img, draw=True)[2] for img in cal_images]
    save_images(cal_chessboards, 'camera_chessboard', ['chessboard_{}.jpg'.format(i) for i in range(len(cal_files))])

    M, dist = calibration.calibrate(cal_images)
    print('Camera matrix:', M)
    print('Distortion coefficients:', dist)
    return cal_images, M, dist

def undistort_images(images, M, dist, folder=None):
    undist_images = [calibration.undistort(img, M, dist) for img in images]
    if folder is not None:
        save_images(undist_images, folder, ['undistorted_{}.jpg'.format(i) for i in range(len(images))])

    return undist_images

def load_test_images(folder='test_images'):
    test_image_filenames = os.listdir(folder)
    print('Found {} test images:\n{}'.format(len(test_image_filenames), '\n'.join(test_image_filenames)))
    test_image_filenames = [os.path.join(folder, fname) for fname in test_image_filenames if fname != '.DS_Store']

    return [calibration.load_image(fname) for fname in test_image_filenames]

def do_sobel(images):
    pairs = (('x', (50, 200), 29), ('y', (70, 255), 15), ('xy', (70, 255), 15))
    for axis , thresh, kernel in pairs:
        s = np.array([sobel.sobel(img, axis=axis, threshold=thresh, kernel=kernel) for img in images])
        save_images(s, 'sobel', ['sobel{}_{}.jpg'.format(axis, i) for i in range(len(images))])
    s = np.array([sobel.sobel(img, directional=True, threshold=(0.9,1.1), kernel=25) for img in images])
    save_images(s, 'sobel', ['sobeld_{}.jpg'.format(i) for i in range(len(images))])

def detect_brightness(image):
    h = image.shape[0]
    bottom = image[h//2:h, :]
    sun = np.zeros_like(bottom)
    sun[bottom > 240] = 1
    bright = np.zeros_like(bottom)
    bright[bottom > 200] = 1
    return sun.sum(), bright.sum(), bottom.mean()

def my_sobel(image):
    m = np.array([
            [-2, 0, 2],
            [-1, 0, 1],
            [-2, 0, 2]
        ])
    return cv2.filter2D(image, -1, m)

def _threshold(image, channel, min_thresh, max_thresh):
    result = np.zeros(image.shape[:2])
    result[(image[...,channel] >= min_thresh) & (image[...,channel] <= max_thresh)] = 1
    return result

def do_thresholding(image, i=None):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    sunny_pixels, bright_pixels, mean = detect_brightness(hls[:,:,2])

    result = None
    min_brightness = 40
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    binary_lab_b = _threshold(lab, 2, 150, 255)
    if i is not None:
        save_images([binary_lab_b], 'threshold', ['{}_bin_lab_b.jpg'.format(i)])
    if sunny_pixels > 15000 or bright_pixels > 15000:
        print('Image {} is sunny with: sunny px = {}, bright px = {}, mean = {}'.format(i, sunny_pixels, bright_pixels, mean))
        # with sun
        # b > 150
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # HSV S > 110
        binary_hsv_s = _threshold(hsv, 1, 110, 255)
        # [sobel x / custom sobel x]
        so = my_sobel(lab[...,0])
        binary_so = np.zeros_like(so)
        binary_so[so > 80] = 1

        result = np.max((binary_lab_b, binary_hsv_s, binary_so), axis=0)
        result[lab[...,0] < 2*min_brightness] = 0
        if i is not None:
            save_images(
                [result, binary_hsv_s, binary_so],
                'threshold',
                ['{}_result_sunny.jpg'.format(i), '{}_bin_hsv_s.jpg'.format(i), '{}_bin_mysobel.jpg'.format(i)]
            )
    elif bright_pixels > 4000:
        print('Image {} is bright with: sunny px = {}, bright px = {}, mean = {}'.format(i, sunny_pixels, bright_pixels, mean))
        # bright n sunny
        # b > 150
        # HLS S > 75 < 255
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        binary_hls_s = _threshold(hls, 2, 90, 255)
        # HSV S > 110
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        binary_hsv_s = _threshold(hsv, 1, 110, 255)
        # R channel > 200
        binary_rgb_r = _threshold(image, 0, 220, 255)

        result = np.max((binary_lab_b, binary_hls_s, binary_hsv_s, binary_rgb_r), axis=0)
        result[lab[...,0] < 2*min_brightness] = 0
        if i is not None:
            save_images(
                [result, binary_hls_s, binary_hsv_s, binary_rgb_r],
                'threshold',
                ['{}_result_bright.jpg'.format(i), '{}_bin_hls_s.jpg'.format(i), '{}_bin_hsv_s.jpg'.format(i), '{}_bin_rgb_r.jpg'.format(i)]
            )
    else:
        print('Image {} is normal with: sunny px = {}, bright px = {}, mean = {}'.format(i, sunny_pixels, bright_pixels, mean))
        # without sun
        # b channel > 150
        # HLS S channel > 75 < 200
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        binary_hls_s = _threshold(hls, 2, 90, 255)
        # HSV S channel > 110
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        binary_hsv_s = _threshold(hsv, 1, 110, 255)
        # R channel > 170?
        binary_rgb_r = _threshold(image, 0, 170, 255) # 200?
        result = np.max((binary_lab_b, binary_hls_s, binary_hsv_s, binary_rgb_r), axis=0)
        result[lab[...,0] < min_brightness] = 0
        if i is not None:
            save_images(
                [result, binary_hls_s, binary_hsv_s, binary_rgb_r],
                'threshold',
                ['{}_result_normal.jpg'.format(i), '{}_bin_hls_s.jpg'.format(i), '{}_bin_hsv_s.jpg'.format(i), '{}_bin_rgb_r.jpg'.format(i)]
            )
    return result

def apply_roi(images, region=None):
    return images


def warp():
    source_points = np.float32((
        (230, 700), (1075, 700), (693, 455), (588, 455)
    ))
    destination_points = np.float32((
        (280, 720), (1000, 720), (1000, 0), (280, 0)
    ))

if __name__ == '__main__':
    chessboards, M, dist = calibrate_camera()
    undistort_images(chessboards, M, dist, 'camera_undistorted')

    images = load_test_images()
    images = undistort_images(images, M, dist, 'test_images_undistorted')

    images = apply_roi(images)
    do_sobel(images)

    thresh_images = [do_thresholding(img, i) for i, img in enumerate(images)]
    save_images(thresh_images, 'threshold', ['binary_{}.jpg'.format(i) for i in range(len(thresh_images))])

    # hand picked subset of test images to showcase some standard and difficult edge cases
    sample_indices = [0, 1, 5, 6, 8, 10, 14, 16, 17, 24]
