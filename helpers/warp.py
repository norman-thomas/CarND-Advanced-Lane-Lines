import cv2
import numpy as np

class Warper:
    def __init__(self):
        self._m = cv2.getPerspectiveTransform(self.source_points, self.destination_points)
        self._minv = cv2.getPerspectiveTransform(self.destination_points, self.source_points)

    @property
    def source_points(self):
        return np.array([
            (230, 700), (1075, 700), (693, 455), (588, 455)
        ], np.float32)

    @property
    def destination_points(self):
        offset = 100
        x1, x2 = 640 - offset, 640 + offset
        return np.array([
            (x1, 720), (x2, 720), (x2, 0), (x1, 0)
        ], np.float32)

    @property
    def M(self):
        return self._m

    @property
    def Minv(self):
        return self._minv
    
    def warp(self, image, newdims=None):
        if newdims is None:
            newdims = image.shape[1], image.shape[0]
        return cv2.warpPerspective(image, self.M, newdims, flags=cv2.INTER_LINEAR)
    
    def unwarp(self, image, newdims=None):
        if newdims is None:
            newdims = image.shape[1], image.shape[0]
        return cv2.warpPerspective(image, self.Minv, newdims, flags=cv2.INTER_LINEAR)
