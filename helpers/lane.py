import numpy as np
import cv2
import math

from collections import namedtuple, deque

from .color import histogram, find_maximum
from .line import Line

class LaneSearch:
    def __init__(self, window_count=12, window_width=60, initial_fraction=4):
        self._image = None
        self._draw_image = None
        self._window_count = window_count
        self._window_width = window_width
        self._initial_fraction = initial_fraction
        self._threshold = 100

        self.left = Line()
        self.right = Line()

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
        return 400

    def search(self, frame, history=None, draw=False):
        self._image = frame
        self._draw_image = np.dstack((frame, frame, frame)) * 255 if draw else None

        result = self._search_smart(draw=draw)
        if result is None:
            result = self._search_dumb(draw=draw)
        if result is None:
            # fallback
            pass
        return result

    def _search_dumb(self, draw=False):
        def _is_tolerable(val, previous_val, skipped=0):
            return val is not None and abs(val - previous_val) <= _margin(skipped)

        def _margin(skipped=0, first_row=False):
            if first_row:
                return 2 * self._window_width
            return int(self._window_width * 1.2**skipped)

        def _draw_rectangle(x, y_from, y_to, skipped=0, first_row=False):
            found = skipped == 0
            delta = _margin(skipped, first_row)
            lim_left = 0 if x < self.image.shape[1]//2 else self.image.shape[1]//2
            lim_right = self.image.shape[1]//2 if x < self.image.shape[1]//2 else self.image.shape[1]
            p1x = max(lim_left, x - delta)
            p2x = min(x + delta, lim_right)
            color = (255, 0, 0) if not found else (0, 255, 0)
            cv2.rectangle(self._draw_image, (p1x, y_from), (p2x, y_to), color=color, thickness=2)

        def _convolve(y1, y2, prev_l, prev_r, l_skipped, r_skipped, first_row=False):
            limit = 150
            height, width = self.image.shape
            left_limit, right_limit = limit, width - limit
            s = np.sum(self.image[y1:y2, :], axis=0)

            def _conv(prev_x: int, skipped: int):
                window_width = _margin(skipped, first_row=first_row)
                surrounding = window_width // 5
                window = np.ones(window_width)
                window[window_width // 2 - 2 * surrounding:window_width // 2 + 2 * surrounding] = 2
                window[window_width // 2 - surrounding:window_width // 2 + surrounding] = 4
                min_index = max(left_limit, prev_x - window_width)
                max_index = min(right_limit, prev_x + window_width)
                c = np.convolve(window, s[min_index:max_index])
                c = c[window_width//2:-window_width//2]
                center = find_maximum(c)
                return center + min_index if c[center] > self._threshold else None

            l_center = _conv(prev_l, l_skipped)
            r_center = _conv(prev_r, r_skipped)

            return l_center, r_center

        height, width = self.image.shape
        from_y = height - (height // self._initial_fraction)
        to_y = height
        from_x, to_x = 0, width
        snip = self.image[from_y:to_y, from_x:to_x]
        hist = histogram(snip)
        margin = 250
        midpoint = len(hist)//2
        left, right = from_x + find_maximum(hist[margin:midpoint]) + margin, from_x + find_maximum(hist[midpoint:width-margin]) + midpoint
        left_centroids = []
        right_centroids = []

        previous_left = left
        previous_right = right

        first_row = True
        left_skipped = 0
        right_skipped = 0
        for row in range(0, self.window_count):
            from_ = height - (row+1) * self.window_height
            to_ = height - row * self.window_height
            center_height = int(height - (row + 0.5) * self.window_height)

            l, r = _convolve(from_, to_, previous_left, previous_right, left_skipped, right_skipped, first_row=first_row)

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

            if (right_skipped + left_skipped) < 2:
                first_row = False

            if draw:
                _draw_rectangle(previous_left, from_, to_, left_skipped, first_row=first_row)
                _draw_rectangle(previous_right, from_, to_, right_skipped, first_row=first_row)

        left_centroids = np.array(left_centroids)
        right_centroids = np.array(right_centroids)
        if draw:
            cv2.polylines(self._draw_image, [left_centroids], False, (0,255,0), thickness=8)
            cv2.polylines(self._draw_image, [right_centroids], False, (0,0,255), thickness=8)

        # calc polynomial
        left_coeffs = self.left.fit(left_centroids)
        right_coeffs = self.right.fit(right_centroids)

        new_left, new_right = self._sanity_check_and_fix(left_coeffs, right_coeffs, left_centroids, right_centroids)
        if new_left is not None:
            self.left.accept_fit(new_left)
        else:
            self.left.reject_fit(left_coeffs)

        if new_right is not None:
            self.right.accept_fit(new_right)
        else:
            self.right.reject_fit(right_coeffs)

        return self.left, self.right

    def _sanity_check_and_fix(self, left_coeffs, right_coeffs, left_centroids, right_centroids, y=720):
        def _cascade(first, second, first_line, second_line, distance):
            result = first
            if result is None:
                last = first_line.last_fit
                if last is not None:
                    result = last.copy()
                else:
                    if second is not None:
                        result = second.copy()
                        result[2] += distance
                    else:
                        last_right = second_line.last_fit
                        if last is not None:
                            result = last_right.copy()
                            result += distance
            return result

        left_coeffs = _cascade(left_coeffs, right_coeffs, self.left, self.right, -self.lane_distance)
        right_coeffs = _cascade(right_coeffs, left_coeffs, self.right, self.left, self.lane_distance)

        if left_coeffs is None or right_coeffs is None:
            return None, None

        result_left, result_right = left_coeffs.copy(), right_coeffs.copy()
        left_func = self.left.create_function(*left_coeffs)
        right_func = self.right.create_function(*right_coeffs)
        left_x0 = left_func(y)
        right_x0 = right_func(y)

        confidence = 1.0

        distance = right_x0 - left_x0
        if distance > 850 or distance < 700:
            confidence *= 0.7

        al, ar = left_coeffs[0], right_coeffs[0]
        #print('al = {}, ar = {}'.format(al, ar))
        a_diff = abs(al - ar)
        a_diff = math.log10(a_diff) if a_diff > 0.0 else -math.inf
        #print('a_diff = {}'.format(a_diff))
        if a_diff > 0.0:
            confidence *= 0.8

        if confidence < 0.7:
            # TODO: maybe use last confident right line instead?
            if len(left_centroids) >= len(right_centroids):
                result_right = left_coeffs.copy()
                result_right[2] += self.lane_distance
            else:
                result_left = right_coeffs.copy()
                result_left[2] -= self.lane_distance
        return result_left, result_right


    @staticmethod
    def _fit_function(xs, ys):
        return np.polyfit(ys, xs, 2)

    def _search_smart(self, draw=False):
        last_left, last_right = self.left.last_fit, self.right.last_fit
        if last_left is None or last_right is None:
            return None

        return None


    def draw_lane(self, image, warper):
        last_left = self.left.last_fit
        last_right = self.right.last_fit
        if last_left is None or last_right is None:
            return image

        func_left = Line.create_function(*last_left)
        func_right = Line.create_function(*last_right)

        ys = np.linspace(0, image.shape[0]-1, image.shape[0]//2)
        xls = np.array([func_left(y) for y in ys])
        xrs = np.array([func_right(y) for y in ys])
        xms = (xrs + xls) / 2

        warp_zero = np.zeros_like(image, dtype=np.uint8)
        color_warp = warp_zero.copy()
        pts_left = np.array([np.transpose(np.vstack([xls, ys]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([xrs, ys])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(0,0,255), thickness = 20)
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        newwarp = warper.unwarp(color_warp)
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

        color_warp = warp_zero.copy()
        pts_center = np.array([np.transpose(np.vstack([xms, ys]))])
        cv2.polylines(color_warp, np.int_([pts_center]), isClosed=False, color=(0,255,255), thickness = 5)

        newwarp = warper.unwarp(color_warp)
        result = cv2.addWeighted(result, 1, newwarp, 0.5, 0)

        return result

