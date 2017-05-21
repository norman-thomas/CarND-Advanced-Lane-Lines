import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

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
        self._previous_distance = None

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
        return 320

    def search(self, frame):
        self._image = frame
        self._draw_image = None

        found = self._smart_sliding_window()
        if not found:
            found = self._sliding_window()
        return found

    def _sliding_window(self):
        frame = self.image
        crop = 250
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(frame[frame.shape[0] // 2:, :], axis=0)
        histogram = histogram[crop:len(histogram)-crop]
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((frame, frame, frame)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint]) + crop
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint + crop

        # Choose the number of sliding windows
        nwindows = self.window_count
        # Set height of windows
        window_height = np.int(frame.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = self._window_width
        # Set minimum number of pixels found to recenter window
        minpix = 50 # self._threshold
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = frame.shape[0] - (window + 1) * window_height
            win_y_high = frame.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if len(leftx) == 0 or len(rightx) == 0:
            left_fit = self.left.current_fit
            right_fit = self.right.current_fit
        else:
            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

        self._draw_image = out_img
        if self._sanity_check(left_fit, right_fit):
            self.left.accept_fit(left_fit)
            self.right.accept_fit(right_fit)
            return True
        #print('DUMB rejected', left_fit, right_fit)
        self.left.reject_fit(left_fit)
        self.right.reject_fit(right_fit)
        return False

    def _smart_sliding_window(self):
        if not self.left.detected or not self.right.detected or self.left.last_fit is None or self.right.last_fit is None:
            return None

        binary_warped = self.image
        left_fit = self.left.last_fit
        right_fit = self.right.last_fit

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = (
            (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
            nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin))
        )
        right_lane_inds = (
            (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
            nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin))
        )

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        if len(leftx) == 0 or len(rightx) == 0:
            return False
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        self._draw_image = result

        if self._sanity_check(left_fit, right_fit):
            self.left.accept_fit(left_fit)
            self.right.accept_fit(right_fit)
            return True
        return False

    def _sanity_check(self, left_coeffs, right_coeffs):
        def _check_first_coeff(v1, v2, t=0.5):
            av1, av2 = abs(v1), abs(v2)
            div = (av1 / av2) if av1 > av2 else (av2 / av1)
            log_div = math.log10(div) if div > 0.0 else -math.inf
            return abs(log_div) < t

        # compare first coefficient
        al, ar = left_coeffs[0], right_coeffs[0]
        if not _check_first_coeff(al, ar):
            return False

        # compare previous fit
        if self.left.detected and self.left.last_fit is not None:
            prev_l0 = self.left.last_fit[0]
            if not _check_first_coeff(prev_l0, al):
                return False
        if self.right.detected and self.right.last_fit is not None:
            prev_r0 = self.right.last_fit[0]
            if not _check_first_coeff(prev_r0, ar):
                return False

        # compare distance
        left_func = Line.create_function(*left_coeffs)
        right_func = Line.create_function(*right_coeffs)
        y = 720
        left_x0 = left_func(y)
        right_x0 = right_func(y)
        if not (320 < abs(left_x0 - right_x0) < 450):
            return False

        return True

    def draw_lane(self, image, warper):
        last_left = self.left.average_fit
        last_right = self.right.average_fit
        if last_left is None or last_right is None:
            return image

        func_left = Line.create_function(*last_left)
        func_right = Line.create_function(*last_right)

        ys = np.linspace(0, image.shape[0]-1, image.shape[0]//2)
        xls = np.array([func_left(y) for y in ys])
        xrs = np.array([func_right(y) for y in ys])
        xms = (xrs + xls) / 2

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3 / 400  # meters per pixel in x dimension

        def off_center(y=720):
            return (func_left(y) + func_right(y)) / 2 - image.shape[1] // 2

        def curvature():
            ploty = ys
            y_eval = np.max(ploty)

            # Fit new polynomials to x,y in world space
            left_fit_cr = np.polyfit(ploty * ym_per_pix, xls * xm_per_pix, 2)
            right_fit_cr = np.polyfit(ploty * ym_per_pix, xrs * xm_per_pix, 2)
            # Calculate the new radii of curvature
            left_curverad = ((1 + (
                2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
            right_curverad = ((1 + (
                2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
                2 * right_fit_cr[0])
            return left_curverad, right_curverad

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

        left_curv, right_curv = curvature()
        curv = (left_curv + right_curv) / 2
        curve_text = "Curvature: {:.2f}m".format(curv)
        font = cv2.FONT_HERSHEY_SIMPLEX
        result = cv2.putText(result, curve_text, (20, 50), font, 1, (255, 255, 255), 2)

        off_center_pixels = off_center()
        off_center_text = "Off center: {:.2f}m".format(off_center_pixels * xm_per_pix)
        result = cv2.putText(result, off_center_text, (20, 80), font, 1, (255, 255, 255), 2)

        return result

