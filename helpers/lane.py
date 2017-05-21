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
        if not found:
            # fallback
            pass
        return found

    def _search_dumb(self):
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

            _draw_rectangle(previous_left, from_, to_, left_skipped, first_row=first_row)
            _draw_rectangle(previous_right, from_, to_, right_skipped, first_row=first_row)

        left_centroids = np.array(left_centroids)
        right_centroids = np.array(right_centroids)

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
            print('DUMB accepted')
            self.left.accept_fit(left_fit)
            self.right.accept_fit(right_fit)
            return True
        print('DUMB rejected', left_fit, right_fit)
        self.left.reject_fit(left_fit)
        self.right.reject_fit(right_fit)
        return False

    def _smart_sliding_window(self):
        if self.left.last_fit is None or self.right.last_fit is None:
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
            print('SMART accepted')
            self.left.accept_fit(left_fit)
            self.right.accept_fit(right_fit)
            return True
        return False

    def _sliding_convolution(self):
        # window settings
        window_width = 50
        window_height = 80  # Break image into 9 vertical layers since image height is 720
        margin = 100  # How much to slide left and right for searching
        warped = self._image

        def window_mask(width, height, img_ref, center, level):
            output = np.zeros_like(img_ref)
            output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
            max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
            return output

        def find_window_centroids(image, window_width, window_height, margin):
            image = self._image

            window_centroids = []  # Store the (left,right) window centroid positions per level
            window = np.ones(window_width)  # Create our window template that we will use for convolutions

            # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
            sums = np.sum(image[image.shape[0] // 2:, :], axis=0)
            # and then np.convolve the vertical image slice with the window template
            conv = np.convolve(sums, window)

            # Sum quarter bottom of image to get slice, could use a different ratio
            l_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)], axis=0)
            l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
            r_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):], axis=0)
            r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(warped.shape[1] / 2)

            # Add what we found for the first layer
            window_centroids.append((l_center, r_center))

            # Go through each layer looking for max pixel locations
            for level in range(1, (int)(warped.shape[0] / window_height)):
                # convolve the window into the vertical slice of the image
                image_layer = np.sum(warped[int(warped.shape[0] - (level + 1) * window_height):int(
                    warped.shape[0] - level * window_height), :], axis=0)
                conv_signal = np.convolve(window, image_layer)
                # Find the best left centroid by using past left center as a reference
                # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
                offset = window_width / 2
                l_min_index = int(max(l_center + offset - margin, 0))
                l_max_index = int(min(l_center + offset + margin, warped.shape[1]))
                l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
                # Find the best right centroid by using past right center as a reference
                r_min_index = int(max(r_center + offset - margin, 0))
                r_max_index = int(min(r_center + offset + margin, warped.shape[1]))
                r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
                # Add what we found for that layer
                window_centroids.append((l_center, r_center))

            return window_centroids

        window_centroids = find_window_centroids(warped, window_width, window_height, margin)

        # If we found any window centers
        if len(window_centroids) > 0:

            # Points used to draw all the left and right windows
            l_points = np.zeros_like(warped)
            r_points = np.zeros_like(warped)

            # Go through each level and draw the windows
            for level in range(0, len(window_centroids)):
                # Window_mask is a function to draw window areas
                l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
                r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
                # Add graphic points from window mask here to total pixels found
                l_points[(l_points == 255) | ((l_mask == 1))] = 255
                r_points[(r_points == 255) | ((r_mask == 1))] = 255

            # Draw the results
            template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
            zero_channel = np.zeros_like(template)  # create a zero color channel
            template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
            warpage = np.array(cv2.merge((warped, warped, warped)),
                               np.uint8)  # making the original road pixels 3 color channels
            output = cv2.addWeighted(warpage, 1, template, 0.5,
                                     0.0)  # overlay the orignal road image with window results

        # If no window centers found, just display orginal road image
        else:
            output = np.array(cv2.merge((warped, warped, warped)), np.uint8)

        # Display the final results
        plt.imshow(output)
        plt.title('window fitting results')
        #plt.show()

        self._draw_image = output

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
        left_func = Line.create_function(*left_coeffs)
        right_func = Line.create_function(*right_coeffs)
        left_x0 = left_func(y)
        right_x0 = right_func(y)

        confidence = 1.0

        tolerance_factor = 1.2
        distance = right_x0 - left_x0
        if distance > self.lane_distance * tolerance_factor or distance < self.lane_distance / tolerance_factor:
            confidence *= 0.7

        al, ar = left_coeffs[0], right_coeffs[0]
        a_diff = abs(al - ar)
        a_diff = math.log10(a_diff) if a_diff > 0.0 else -math.inf
        #print('distance = {}, al = {}, ar = {}, a_diff = {}, a_div = {}'.format(distance, al, ar, a_diff, math.log10(abs(al/ar))))
        if a_diff > 0.0:
            confidence *= 0.8
        #elif a_diff < 0.0:
        #    new_a = (len(left_centroids) * al + len(right_centroids) * ar) / (len(left_centroids) + len(right_centroids))
        #    result_left[0] = new_a
        #    result_right[0] = new_a

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

    def _search_smart(self):
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

