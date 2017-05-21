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
import helpers.lane as lane
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

def prepare(recreate=False):
    PICKLE = 'temp.p'
    camera = None
    if recreate or not os.path.exists(PICKLE):
        print('Calibrating camera...')
        camera = calibrate_camera()
        #undistort_images(chessboards, M, dist, 'camera_undistorted')
        print('Saving pickle with calibration info...')
        with open(PICKLE, 'wb') as f:
            pickle.dump((camera.M, camera.dist), f)
    else:
        with open(PICKLE, 'rb') as f:
            M, dist = pickle.load(f)
        camera = Camera(M, dist)
    return camera

def prepare_images(camera, recreate=False):
    images = None
    if recreate or not os.path.exists(os.path.join(OUTPUT_FOLDER, 'test_images_undistorted', 'undistorted_00.jpg')):
        print('Undistorting test images...')
        images = load_images('test_images')
        images = undistort_images(images, camera, 'test_images_undistorted')
    else:
        images = load_images(os.path.join(OUTPUT_FOLDER, 'test_images_undistorted'))

    thresh_images = None
    if recreate or not os.path.exists(os.path.join(OUTPUT_FOLDER, 'threshold', 'binary_00.jpg')):
        print('Looking for edges and applying color thresholds...')
        #do_sobel(images)
        thresh_images = [color.ColorThreshold.threshold(img) for i, img in enumerate(images)]
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

    return images, warped_binaries

def process(camera, warper, s):
    def _process(img):
        img = camera.undistort(img)
        thresh = color.ColorThreshold.threshold(img)
        warped = warper.warp(thresh)
        funcs = s.search(warped)
        if funcs is not None:
            return s.draw_lane(img, warper)
        return img
    return _process

def main_image():
    camera = prepare()
    images, warped_binaries = prepare_images(camera, recreate=False)
    warper = Warper()

    for i, img in enumerate(warped_binaries[:15]):
        print('>>> Searching for lanes in image {}...'.format(i))
        s = lane.LaneSearch(window_count=15)
        l, r = s.search(img, draw=True)
        print('\t', i, 'left coeffs:', l.coeffs)
        print('\t', i, 'right coeffs:', r.coeffs)
        save_images([s.draw_image], 'draw', ['draw_{:02d}.jpg'.format(i)])
        hud = s.draw_lane(images[i], warper)
        save_images([hud], 'hud', ['hud_{:02d}.jpg'.format(i)])

def main_video():
    camera = prepare()
    warper = Warper()
    s = lane.LaneSearch(window_count=15)
    clip = VideoFileClip('project_video.mp4')
    clip = clip.subclip(t_start=39.5, t_end=42.5)
    result = clip.fl_image(process(camera, warper, s))
    result.write_videofile('out.mp4', audio=False)


if __name__ == '__main__':
    main_image()
    #main_video()
