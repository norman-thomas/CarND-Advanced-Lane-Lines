import os
#import glob
import pickle

import numpy as np
import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip

import random
import math

import matplotlib.image as mpimg

from helpers.calibration import Camera
from helpers.warp import Warper
import helpers.sobel as sobel
import helpers.color as color
import helpers.line as line
import helpers.utils as utils

OUTPUT_FOLDER = 'output_images'

def save_images(images, folder, filenames):
    path = os.path.join(OUTPUT_FOLDER, folder)
    if not os.path.exists(path):
        os.mkdir(path)
    for img, fname in zip(images, filenames):
        if len(img.shape) == 2:
            mpimg.imsave(os.path.join(path, fname), img, format='jpg', cmap='gray')
        else:
            mpimg.imsave(os.path.join(path, fname), img, format='jpg')

def calibrate_camera(folder='camera_cal'):
    cal_files = os.listdir(folder)
    cal_files = list(map(lambda f: os.path.join(folder, f), cal_files))
    cal_images = [ utils.load_image(f) for f in cal_files ]
    print('There are {} calibration images present'.format(len(cal_files)))

    camera = Camera()
    M, dist = camera.calibrate(cal_images)
    print('Camera matrix:', M)
    print('Distortion coefficients:', dist)
    return camera

def undistort_images(images, camera, folder=None):
    undist_images = [camera.undistort(img) for img in images]
    if folder is not None:
        save_images(undist_images, folder, ['undistorted_{:02d}.jpg'.format(i) for i in range(len(images))])
    return undist_images

def load_images(folder, gray=False, debug=False):
    test_image_filenames = os.listdir(folder)
    print('Loading {} images from {}'.format(len(test_image_filenames), folder))
    if debug:
        print('\n'.join(test_image_filenames))
    test_image_filenames = [os.path.join(folder, fname) for fname in test_image_filenames if fname != '.DS_Store']

    return [utils.load_image(fname, gray=gray) for fname in test_image_filenames]

def do_sobel(images):
    pairs = (('x', (50, 200), 29), ('y', (70, 255), 15), ('xy', (70, 255), 15))
    for axis , thresh, kernel in pairs:
        s = np.array([sobel.sobel(img, axis=axis, threshold=thresh, kernel=kernel) for img in images])
        save_images(s, 'sobel', ['sobel{}_{:02d}.jpg'.format(axis, i) for i in range(len(images))])
    s = np.array([sobel.sobel(img, directional=True, threshold=(0.9,1.1), kernel=25) for img in images])
    save_images(s, 'sobel', ['sobeld_{:02d}.jpg'.format(i) for i in range(len(images))])

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
        #print('Image {} is sunny with: sunny px = {}, bright px = {}, mean = {}'.format(i, sunny_pixels, bright_pixels, mean))
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
        #print('Image {} is bright with: sunny px = {}, bright px = {}, mean = {}'.format(i, sunny_pixels, bright_pixels, mean))
        # bright n sunny
        # ~20 < L < 50 && b > 120 || L > 50 && b > 150
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
        result[(lab[...,0] > 20) & (lab[...,0] < 50) & (lab[...,2] > 120)] = 1
        if i is not None:
            save_images(
                [result, binary_hls_s, binary_hsv_s, binary_rgb_r],
                'threshold',
                ['{}_result_bright.jpg'.format(i), '{}_bin_hls_s.jpg'.format(i), '{}_bin_hsv_s.jpg'.format(i), '{}_bin_rgb_r.jpg'.format(i)]
            )
    else:
        #print('Image {} is normal with: sunny px = {}, bright px = {}, mean = {}'.format(i, sunny_pixels, bright_pixels, mean))
        # without sun
        # b channel > 150
        # HLS S channel > 75 < 200
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        binary_hls_s = _threshold(hls, 2, 90, 255)
        # HSV S channel > 110
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        binary_hsv_s = _threshold(hsv, 1, 110, 255)
        # R channel > 170?
        binary_rgb_r = _threshold(image, 0, 200, 255) # 170 or 200?
        result = np.max((binary_lab_b, binary_hls_s, binary_hsv_s, binary_rgb_r), axis=0)
        result[lab[...,0] < min_brightness] = 0
        if i is not None:
            save_images(
                [result, binary_hls_s, binary_hsv_s, binary_rgb_r],
                'threshold',
                ['{}_result_normal.jpg'.format(i), '{}_bin_hls_s.jpg'.format(i), '{}_bin_hsv_s.jpg'.format(i), '{}_bin_rgb_r.jpg'.format(i)]
            )
    return result

def apply_roi(image):
    points = np.array([[
        [0, 720],
        [0, 670],
        [580, 455],
        [700, 455],
        [1280, 670],
        [1280, 720]
    ]])
    white = 255
    if len(image.shape) > 2:
        white = (white,) * image.shape[2]

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, points, white)
    return cv2.bitwise_and(image, mask)

PICKLE = 'temp.p'
def prepare(recreate=False):
    camera = None
    if recreate or not os.path.exists(PICKLE):
        print('Calibrating camera...')
        camera = calibrate_camera()
        #undistort_images(chessboards, M, dist, 'camera_undistorted')
        print('Saving pickle with calibration info...')
        with open(PICKLE, 'wb') as f:
            pickle.dump((M, dist), f)
    else:
        with open(PICKLE, 'rb') as f:
            M, dist = pickle.load(f)

    images = None
    if recreate or not os.path.exists(os.path.join(OUTPUT_FOLDER, 'test_images_undistorted', 'undistorted_00.jpg')):
        print('Undistorting test images...')
        images = load_images('test_images')
        images = undistort_images(images, camera, 'test_images_undistorted')
    else:
        images = load_images(os.path.join(OUTPUT_FOLDER, 'test_images_undistorted'))

    #images = [apply_roi(img) for img in images]
    thresh_images = None
    if recreate or not os.path.exists(os.path.join(OUTPUT_FOLDER, 'threshold', 'binary_00.jpg')):
        print('Looking for edges and applying color thresholds...')
        #do_sobel(images)
        thresh_images = [do_thresholding(img) for i, img in enumerate(images)]
        save_images(thresh_images, 'threshold', ['binary_{:02d}.jpg'.format(i) for i in range(len(thresh_images))])
    else:
        thresh_images = load_images(os.path.join(OUTPUT_FOLDER, 'threshold'), gray=True)

    warped_binaries = None
    if recreate or not os.path.exists(os.path.join(OUTPUT_FOLDER, 'warped_binary', 'warped_binary_00.jpg')):
        warper = Warper()
        print('Warping images...')
        warped_images = [warper.warp(img) for img in images]
        warped_binaries = [warper.warp(img) for img in thresh_images]
        save_images(warped_images, 'warped', ['warped_{:02d}.jpg'.format(i) for i in range(len(warped_images))])
        save_images(warped_binaries, 'warped_binary', ['warped_binary_{:02d}.jpg'.format(i) for i in range(len(warped_binaries))])
    else:
        warped_binaries = load_images(os.path.join(OUTPUT_FOLDER, 'warped_binary'), gray=True)

    return images, warped_binaries, M, dist

def process(camera, warper, s):
    def _process(img):
        img = camera.undistort(img)
        thresh = do_thresholding(img)
        warped = warper.warp(thresh)
        funcs = s.search(warped)
        line.draw_lane(img, *funcs, warper)
        return img
    return _process


if __name__ == '__main__':
    images, warped_binaries, camera_M, camera_dist = prepare()
    camera = Camera(camera_M, camera_dist)
    warper = Warper()

    for i, img in enumerate(warped_binaries[:14]):
        print('Searching for lanes in image {}...'.format(i))
        s = line.LaneSearch(window_count=9, window_width=100)
        s.search(img, draw=True)
        save_images([s.draw_image], 'draw', ['draw_{:02d}.jpg'.format(i)])

    #s = line.LaneSearch(window_count=9, window_width=100)
    #clip = VideoFileClip('project_video.mp4')
    #M, Minv = get_matrices()
    #result = clip.fl_image(process(camera, warper, s,))
    #result.write_videofile('out.mp4', audio=False)


