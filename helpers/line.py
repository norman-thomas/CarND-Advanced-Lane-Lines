import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

from collections import namedtuple
from scipy.optimize import curve_fit

from .color import histogram, find_maximum

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


Lane = namedtuple('Lane', ['num', 'detected', 'coeffs', 'func'])
class LaneClass:
    def __init__(self):
        self.num = -1
        self.detected = False
        self.coeffs = None

def circle(x, xc, yc, r):
    return np.sqrt(r*r - (x-xc)*(x-xc)) + yc

def quadratic(x, a, b, c):
    return a*(x + b)**2 + c

def _create_func(a, b, c):
    def _quadratic(x):
        return quadratic(x, a, b, c)
    return _quadratic

def _transform_quadratic(a, b, c):
    '''transforms the polynomial a*x^2 + b*x + c into the form a * (x + b)^2 + c'''
    return (a, b/(2*a), (c - b**2/(4*a)))

class LaneSearch:

    def __init__(self, window_count=9, window_width=100, initial_fraction=4):
        self._image = None
        self._draw_image = None
        self._initial_fraction = initial_fraction
        self._window_count = window_count
        self._window_width = window_width
        self._history = []
        self._threshold = 300
        self._count = 0

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

    @property
    def lane_distance(self):
        return 725

    def search(self, frame, history=None, draw=False):
        self._image = frame
        self._draw_image = np.dstack((frame, frame, frame)) * 255 if draw else None
        
        result = self._search_smart(history, draw=draw)
        if result is None:
            result = self._search_dumb(draw=draw)
        if result is None:
            # fallback
            pass
        return result

    def _search_dumb(self, draw=False):
        def _is_tolerable(val, previous_val, skipped):
            return val is not None and abs(val - previous_val) <= _tolerance(skipped)

        def _tolerance(skipped, margin=None):
            return (1+skipped) * (self._window_width//2 if margin is None else margin)

        def _draw_rectangle(x, y_from, y_to, found=True, delta=None):
            delta = delta if delta is not None else self._window_width//2
            lim_left = 0 if x < self.image.shape[1]//2 else self.image.shape[1]//2
            lim_right = self.image.shape[1]//2 if x < self.image.shape[1]//2 else self.image.shape[1]
            p1x = max(lim_left, x - delta)
            p2x = min(x + delta, lim_right)
            color = (255, 0, 0) if not found else (0, 255, 0)
            cv2.rectangle(self._draw_image, (p1x, y_from), (p2x, y_to), color=color, thickness=6)

        def _convolve(y1, y2, prev_l, prev_r, l_margin, r_margin):
            height, width = self.image.shape
            offset = self._window_width//2
            s = np.sum(self.image[y1:y2, :], axis=0)
            window = np.ones(self._window_width)
            surrounding = self._window_width // 5
            window[self._window_width//2-2*surrounding:self._window_width//2+2*surrounding] = 3
            window[self._window_width//2-surrounding:self._window_width//2+surrounding] = 5
            conv = np.convolve(window, s)

            l_min_index = int(max(prev_l+offset-l_margin, 0))
            l_max_index = int(min(prev_l+offset+l_margin, width))
            conv_win = conv[l_min_index:l_max_index]
            l_center = find_maximum(conv_win)
            l_center = l_center + l_min_index - offset if conv_win[l_center] > self._threshold else None

            r_min_index = int(max(prev_r+offset-r_margin, 0))
            r_max_index = int(min(prev_r+offset+r_margin, width))
            conv_win = conv[r_min_index:r_max_index]
            r_center = find_maximum(conv_win)
            r_center = r_center + r_min_index - offset if conv_win[r_center] > self._threshold else None

            return l_center, r_center


        height = self.image.shape[0]
        from_ = height - (height // self._initial_fraction)
        to_ = height
        hist = histogram(self.image, from_=from_)
        left, right = find_maximum(hist[:len(hist)//2]), find_maximum(hist[len(hist)//2:]) + len(hist)//2
        left_centroids = [(left, height)]
        right_centroids = [(right, height)]

        previous_left = left
        previous_right = right
        if draw:
            _draw_rectangle(previous_left, height - self.window_height, to_)
            _draw_rectangle(previous_right, height - self.window_height, to_)

        left_skipped = 0
        right_skipped = 0
        dy = self.window_height//2
        for row in range(1, self.window_count):
            from_ = height - (row+1) * self.window_height
            to_ = height - row * self.window_height
            center_height = int(height - (row + 0.5) * self.window_height)

            margin = 100
            left_margin = (1+left_skipped) * margin
            right_margin = (1+right_skipped) * margin
            l, r = _convolve(from_, to_, previous_left, previous_right, left_margin, right_margin)

            if _is_tolerable(l, previous_left, left_skipped):
                left_skipped = 0
                previous_left = l
                left_centroids.append((l, center_height))
            else:
                left_skipped += 1

            if _is_tolerable(r, previous_right, right_skipped):
                right_skipped = 0
                previous_right = r
                right_centroids.append((r, center_height))
            else:
                right_skipped += 1

            if draw:
                delta = _tolerance(left_skipped, margin)
                _draw_rectangle(previous_left, from_, to_, left_skipped == 0, delta=delta)
                delta = _tolerance(right_skipped, margin)
                _draw_rectangle(previous_right, from_, to_, right_skipped == 0, delta=delta)

        left_centroids = np.array(left_centroids)
        right_centroids = np.array(right_centroids)
        if draw:
            cv2.polylines(self._draw_image, [left_centroids], False, (0,255,0), thickness=15)
            cv2.polylines(self._draw_image, [right_centroids], False, (0,0,255), thickness=15)

        # calc polynomial
        funcs = self._fit(left_centroids, right_centroids)

        self._count += 1
        return funcs

    def _fit(self, left_centroids, right_centroids):
        f_l, f_r = None, None
        if len(left_centroids) >= 3:
            left_xs = [p[0] for p in left_centroids]
            left_ys = [p[1] for p in left_centroids]
            left_func = self._fit_function(left_xs, left_ys)
            f_l = _transform_quadratic(*left_func) if left_func is not None else None
        if len(right_centroids) >= 3:
            right_xs = [p[0] for p in right_centroids]
            right_ys = [p[1] for p in right_centroids]
            right_func = self._fit_function(right_xs, right_ys)
            f_r = _transform_quadratic(*right_func) if right_func is not None else None

        print('f_l = {}\nf_r = {}'.format(f_l, f_r))

        left = None
        right = None
        confidence = 1
        if f_l is not None and f_r is not None:
            horizontal_space = f_r[-1] - f_l[-1]
            if not(700 < horizontal_space < 850):
                confidence *= 0.8

            # compare the scale of the quadratic coefficients
            p1c2 = math.log10(abs(f_l[0]))
            p2c2 = math.log10(abs(f_r[0]))
            coeff_diff = p1c2 / p2c2
            if not(0.75 < coeff_diff < 1.25):
                confidence *= 0.7

            if confidence < 0.7:
                # stronger preference for side with more valid entries
                if len(left_centroids) >= len(right_centroids):
                    f_r = list(f_l)
                    f_r[2] += self.lane_distance
                    f_r = tuple(f_r)
                else:
                    f_l = list(f_r)
                    f_l[2] -= self.lane_distance
                    f_l = tuple(f_l)
            left = Lane(self._count, True, f_l, _create_func(*f_l))
            right = Lane(self._count, True, f_r, _create_func(*f_r))
        elif f_l is not None:
            f_r = list(f_l)
            f_r[2] += self.lane_distance
            f_r = tuple(f_r)
            left = Lane(self._count, True, f_l, _create_func(*f_l))
            right = Lane(self._count, False, f_r, _create_func(*f_r))
        elif f_r is not None:
            f_l = list(f_r)
            f_l[2] -= self.lane_distance
            f_l = tuple(f_l)
            left = Lane(self._count, False, f_l, _create_func(*f_l))
            right = Lane(self._count, True, f_r, _create_func(*f_r))
        else:
            left, right = self._find_last_detected()
            left = Lane(self._count, False, left.coeffs, left.func)
            right = Lane(self._count, False, right.coeffs, right.func)

        self._history.append((left, right))
        left, right = self._find_last_detected()
        return left.func, right.func

    def _find_last_detected(self):
        r = []
        for i in range(2):
            last = list(filter(lambda f: f[i].detected, self._history))[-1]
            r.append(last[i])
        return r


    @staticmethod
    def _fit_function(xs, ys):
        return np.polyfit(ys, xs, 2)

    @staticmethod
    def _fit_circle(xs, ys):
        return curve_fit(circle, ys, xs)


    def _search_smart(self, history, draw=False):
        pass

