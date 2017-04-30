import cv2
import numpy as np

def load_image(fname, debug=False):
    img = np.array(cv2.imread(fname))
    if debug:
        print('Opened file {}, with shape {}'.format(fname, img.shape))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def chessboard(img, n=(9, 6), draw=False):
    gray = img
    if len(img.shape) == 3:
        gray = rgb2gray(img)
    ret, corners = cv2.findChessboardCorners(gray, n, None)
    if ret and draw:
        chess_img = img.copy()
        cv2.drawChessboardCorners(chess_img, n, corners, ret)
        return ret, corners, chess_img
    return ret, corners, img

def calibrate(images, n=(9, 6)):
    imgpoints = []
    for image in images:
        ret, corners, _ = chessboard(image, n=n)
        if ret:
            imgpoints.append(corners)

    objp = np.zeros((np.prod(n), 3), np.float32)
    objp[:, :2] = np.mgrid[0:n[0], 0:n[1]].T.reshape(-1, 2)
    objpoints = [objp] * len(imgpoints) # same length required by OpenCV

    if len(imgpoints) > 0:
        shape = images[0].shape[:2][::-1]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
        if ret:
            return mtx, dist
    return None, None

def undistort(img, M, dist):
    return cv2.undistort(img, M, dist, None, M)
