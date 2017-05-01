import cv2
import numpy as np

class Camera:
    def __init__(self, m=None, dist=None):
        self.M = m
        self.dist = dist
        self._chessboards = []
    
    @staticmethod
    def _chessboard(img, dims=(9, 6), draw=False):
        gray = img
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, dims, None)
        if ret and draw:
            chess_img = img.copy()
            cv2.drawChessboardCorners(chess_img, dims, corners, ret)
            return ret, corners, chess_img
        return ret, corners, img

    def calibrate(self, images, dims=(9, 6), draw=False):
        imgpoints = []
        for image in images:
            ret, corners, rimg = self._chessboard(image, dims=dims, draw=draw)
            if ret and draw:
                self._chessboards.append(rimg)
            if ret:
                imgpoints.append(corners)

        objp = np.zeros((np.prod(dims), 3), np.float32)
        objp[:, :2] = np.mgrid[0:dims[0], 0:dims[1]].T.reshape(-1, 2)
        objpoints = [objp] * len(imgpoints) # same length required by OpenCV

        if len(imgpoints) > 0:
            shape = images[0].shape[:2][::-1]
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
            if ret:
                self.M = mtx
                self.dist = dist
                return mtx, dist
        return None, None

    def undistort(self, img):
        return cv2.undistort(img, self.M, self.dist, None, self.M)
