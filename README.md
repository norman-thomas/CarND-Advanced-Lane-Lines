# CarND Project 4: Advanced Lane Lines

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)


[calibration1]: ./camera_cal/calibration2.jpg "Distorted"
[calibration2]: ./output_images/camera_chessboard/chessboard_11.jpg "Chessboard"
[calibration3]: ./output_images/camera_undistorted/undistorted_11.jpg "Undistorted"

[original1]: ./test_images/test1.jpg "Distorted"
[undistorted1]: ./output_images/test_images_undistorted/undistorted_02.jpg "Undistorted"
[threshold1]: ./output_images/threshold/binary_02.jpg "Thresholded"


### [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

## Camera Calibration

### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

When taking images with a camera, the lens mapping the 3D world to a 2D image causes distortions. Distortions interfere with geometric calculations based on images. Therefore it is necessary to determine the lens' distortion in order to be able to neutralize it. To do that, I use the provided chessboard images taken with the camera used for the images and videos in this project. My implementation of the chessboard detection and camera calibration is located in [helpers/calibration.py](helpers/calibration.py). It uses `cv2.findChessboardCorners(...)` to find the coordinates of chessboard intersections and calculateds the transformation matrix to undistort the image via `cv2.calibrateCamera(...)`. The resulting matrix is then used to apply `cv2.undistort(...)` and undistort an image.

| Original out of camera image | Detected chessboard | Undistorted image |
|:---:|:---:|:---:|
| ![Original distorted image][calibration1] | ![Original image with chessboard][calibration2] | ![Undistorted image][calibration3] |


## Pipeline (single images)

### 1. Provide an example of a distortion-corrected image.

Here's an example of the above distortion correction being applied to a road image from the video:

| Original image | Undistorted image |
|:---:|:---:|
| ![Original image][original1] | ![Undistorted image][undistorted1] |


### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I tried out several color thresholds and Sobel filters as well as combinations of both. All implementations I tried can be found in [helpers/color.py, line 52 and following](helpers/color.py#L52). It seemed that pure color thresholds yield more reliable results as edge detection often detects edges irrelevant to lane finding and therefore confuses the algorithm during further processing. I chose to use the color spaces HLS, HSV and Lab for color thresholding. The final implementation is called via [`ColorThreshold.threshold(...)`](helpers/color.py#L57), which in turn ends up calling [`ColorThreshold._simple_threshold(...)`](helpers/color.py#L88).


| Undistorted image | Thresholded image |
|:---:|:---:|
| ![Undistorted image][undistorted1] | ![Thresholded image][threshold1] |

### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

## Pipeline (video)

### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

## Discussion

### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

