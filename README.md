---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # "Image References"

[image1]: ./examples/undistort_output.jpg "Undistorted"
[image2]: ./examples/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---

### 

### Camera Calibration

The code for this step is contained in a "Step 1: Camera Calibration" section  located in the "adv_lane_lines.py"  at lines 38 through 74.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images and images extracted from the project video. First the calibration Matrix along with the distortion coefficients are calculated in step #1 base on the chessboard patterns, subsequently all images are process by a undistort() function located at line  #81. The  undistort() function uses the calibration matrix  and distortion coefficients to correct images. Below is an example of this process:![alt text][image2]

#### 2. Methods to create a thresholded binary image. 

I used a combination of color and gradient thresholds and a region of interest (ROI) mask to generate a binary image. The threshold function in implemented on lines from 163 through 286. I experimented with many different option finally settling on a threshold implementation based on combination of (LUV) L binary component combined with  sobel x binary component calculated on a gray image. ROI mask is later applied to the combined binary image.  Here's an example of my output for this step. The actual output is marked "9.OUTPUT: #8 masked" in the picture below. I experimented also with the (HLS) S component but although it provided good results detecting lines on the concrete sections of the road it generated a lot of unwanted "noise" on the sections with shadow. The LUV L component didn't provide as good results on the concrete sections but was very resilient to the sections with shadows. That is why for the final solution I selected combination of the "luv_l_binary" and  the "gray_sobelx_binary" images.![alt text][image3]

#### 3. Perspective transform

The code for my perspective transform includes a function called `warp()`, which appears in lines 320 through 252 in the `adv_lane_lines.py file.`As an input the `warp()` function takes an image (`img`), as well as perspective matrix M (global variable) that was calculated based on the source and destination points in the following manner lines 298-312 :

```python
s1 = [595,450]
s2 = [684,450]
s3 = [1108,720]
s4 = [202,s3[1]]
d1 = [s4[0]+coef,0]
d2 = [s3[0]-coef,0]
d3 = [s3[0]-coef,s3[1]]
d4 = [s4[0]+coef,s4[1]]
src = np.float32([s1, s2, s3, s4])
dst = np.float32([d1, d2, d3, d4])
```

This resulted in the following source and destination points:

|  Source  | Destination |
| :------: | :---------: |
| 595, 450 |   282, 0    |
| 684, 450 |   1028, 0   |
| 1108,720 |  1028,720   |
| 202,720  |   282,720   |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Lane-line pixels fitting with a polynomial

I use a convolution base function that calculates centroids to determine line location in the picture. The find_window_centroids() function used to perform this operation is located at line 398 in the "adv_line_lanes.py" file. I extended the original find_window_centroids() function() presented in the lab to be more robust and handle situation where the left and right lines are falsely detected or not detected at all. The output of the find_window_centroids() function is later used in the image_processing_pipeline() function at lines 633-648  to calculate line line parameters that are later used at line 667 to generate an image shown below:

![alt text][image5]

#### 5. Radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature and position of vehicle is calculated in the calc_curvature_pos() function at lines 562 through 587.

#### 6. Example image of result plotted back down onto the road.

I implemented this step in  the image_processing_pipeline() function at lines 672 through 686. The lines are first drawn onto  the warped blank image. The image is than transformed-back(unwarped) to the original image space using inverse perspective matrix (Minv) and combined with the original undistorted image. Here is an example of my result on test images:

![alt text][image6]

---

### Pipeline (video)

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

One of the problems I noticed with my implementation is stability of the lines especially on the concrete sections of the road. The (LUV) L component doesn't provide as good results as (HLS) S components so perhaps combining both and resolving the (HLS) S problems in the sections with shadows can improve results.