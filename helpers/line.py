import numpy as np
import cv2
import math

from .color import histogram, find_maximums

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None


def circle(x, xc, yc, r):
    return math.sqrt(r**2 - (x-xc)**2) + yc

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

class LaneSearch:

    def __init__(self, window_count=9, window_width=100, initial_fraction=4):
        self._initial_fraction = initial_fraction
        self._window_count = window_count
        self._window_width = window_width
        self._history = []

    @property
    def image(self):
        return self._image

    @property
    def draw_image(self):
        return self._draw_image

    @property
    def window_count(self):
        return self._window_count

    @property
    def window_height(self):
        return self.image.shape[0] // self.window_count

    @property
    def window_size(self):
        return self.window_height, self._window_width

    def search(self, frame, smart=False, history=None, draw=False):
        self._image = frame
        self._draw_image = np.dstack((frame, frame, frame)) * 255 if draw else None
        if smart:
            return self._search_smart(history, draw=draw)
        else:
            return self._search_dumb(draw=draw)

    def _draw_rect(corners):
        pass

    def _draw_line(points):
        pass

    def _search_dumb(self, draw=False):
        def _is_tolerable(val, previous_val, skipped):
            return abs(val - previous_val) <= (1+skipped) * self._window_width//2

        height = self.image.shape[0]
        hist = histogram(self.image, from_=int(height * (1 - 1/self._initial_fraction)))
        left, right = find_maximums(hist)
        windows = []
        left_centroids = [(left, height)]
        right_centroids = [(right, height)]

        previous_left = left
        previous_right = right

        left_skipped = 0
        right_skipped = 0
        for row in range(1, self.window_count):
            from_ = height - (row+1) * self.window_height
            to_ = height - row * self.window_height
            section = self.image[from_:to_]
            hist = histogram(section)
            left, right = find_maximums(hist)
            if _is_tolerable(left, previous_left, left_skipped):
                left_skipped = 0
                previous_left = left
                left_centroids += [(left, height - row * self.window_height)]
            else:
                left_skipped += 1
            if _is_tolerable(right, previous_right, right_skipped):
                right_skipped = 0
                previous_right = right
                right_centroids += [(right, height - row * self.window_height)]
            else:
                right_skipped += 1

        left_centroids = np.array(left_centroids)
        right_centroids = np.array(right_centroids)
        if draw:
            cv2.polylines(self._draw_image, [left_centroids], False, (0,255,0), thickness=15)
            cv2.polylines(self._draw_image, [right_centroids], False, (0,0,255), thickness=15)

        # TODO self._history.append()
        return left_centroids, right_centroids


    def _search_smart(self, history, draw=False):
        pass

