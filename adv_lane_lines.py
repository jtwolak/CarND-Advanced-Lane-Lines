import pickle
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

title_font = {'fontname':'Arial', 'size':'14', 'color':'black', 'weight':'normal', 'verticalalignment':'bottom'}
title_font_sm = {'fontname':'Arial', 'size':'12', 'color':'black', 'weight':'normal', 'verticalalignment':'bottom'}
label_font = {'fontname':'Arial', 'size':'8'}
axis_font = {'fontname':'Arial', 'size':'8'}

threshold_show_results = "NO"
undistort_show_results = "NO"
warp_show_results = "NO"
image_preprocess_show_results = "NO"
image_processing_pipeline_show_line_filling = "NO"

process_test_images = "NO"
process_test_images_show_results = "NO"
process_video = "YES"
process_video_show_results = "NO"
process_video_dump_imgs = "NO"

testImageSet = []
# Load the test images to show various operations
testImageFileNames = glob.glob('test_images/test*.jpg')
#testImageFileNames = ['test/img_in75.jpg', 'test/img_in86.jpg', 'test/img_in87.jpg', 'test/img_in88.jpg', 'test/img_in89.jpg', 'test/img_in90.jpg']

for fname in testImageFileNames:
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    testImageSet.append(img)

#===================================================================
# Step 1: Camera Calibration
#===================================================================
# Read in calibration corner points
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Read in calibration images of a chessboard
images = glob.glob('camera_cal/calibration*.jpg')
# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9, 6), corners, ret)

# Calculate distortion coefficients and the camera matrix
#NOTE: We need the distortion coefficients and the camera matrix to undistort images.
#      They are used to map 3D object points to 2D image points
# At the same time test the calibration and distortion on one test image selected from
# the chessboard images.
def calibrate(img, objpoints, imgpoints):
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return mtx, dist

test_img = cv2.imread(images[15])
mtx, dist = calibrate(test_img, objpoints, imgpoints)

#===================================================================
# Step 2: Define the Distortion Correction Function
# It uses distortion coefficients and the camera matrix computed
# in the step #1
#===================================================================
def undistort( image, show_results="NO" ):
    undistorted = cv2.undistort(image, mtx, dist, None, mtx)
    if show_results == "YES":
        # Show the undistorted test image
        f, (ax1, ax2) = plt.subplots(1, 2)
        plt.axis('off')
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        ax2.imshow(undistorted)
        ax2.set_title('Corrected(Undistorted) Image')
        ax2.axis('off')
        plt.show()
    return undistorted

if undistort_show_results == "YES":
    undistort( test_img, "YES" )
    undistort( testImageSet[0], "YES" )

#===================================================================
# Step 3. Implementation of color and gradient threshold
# Performs the camera calibration, image distortion correction and
# returns the undistorted image
#====================================================================
def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

#=============================================================
# Calculates gradient direction
# Applies threshold
#==============================================================
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    dir_grad = np.arctan2(abs_sobely, abs_sobelx)

    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(dir_grad)
    binary_output[(dir_grad >= thresh[0]) & (dir_grad <= thresh[1])] = 1

    # 6) Return this mask as your binary_output image
    # binary_output = np.copy(img) # Remove this line
    return binary_output

#=============================================================
# Threshold function that applies Sobel in x and y and/or
# threshold operation in HLS color space that makes lines stand out
# As an output it generates binary image.
# Last BKC: s_thresh=(170, 255), sx_thresh=(20, 100)
#==============================================================
def threshold(img, show_results="NO", s_thresh=(150, 255), sx_thresh=(25, 120)):
    img_rgb = np.copy(img)

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS).astype(np.float)
    hls_h = hls[:, :, 0]
    hls_l = hls[:, :, 1]
    hls_s = hls[:, :, 2]

    # Calculate Grayscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float)

    # Convert RGB image to LUV format
    luv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LUV).astype(np.float)
    luv_l = luv[:, :, 1]
    luv_l_binary = np.uint8( np.zeros_like(luv_l) )
    luv_l_binary[(luv_l >= 115) & (luv_l <= 255)] = 1
    luv_l_binary_show =  luv_l_binary * 255

    # Sobel X
    gray_sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    gray_sobelx_abs = np.absolute(gray_sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    gray_sobelx_scaled = np.uint8(255 * gray_sobelx_abs / np.max(gray_sobelx_abs))
    gray_sobelx_binary = np.zeros_like(gray_sobelx_scaled)
    # Apply threshold and generate binary image
    gray_sobelx_binary[(gray_sobelx_scaled >= sx_thresh[0]) & (gray_sobelx_scaled <= sx_thresh[1])] = 1
    gray_sobelx_binary_show = gray_sobelx_binary * 255

    # Sobel Y
    gray_sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)  # Take the derivative in x
    gray_sobely_abs = np.absolute(gray_sobely)  # Absolute x derivative to accentuate lines away from horizontal
    gray_sobely_scaled = np.uint8(255 * gray_sobely_abs / np.max(gray_sobely_abs))
    gray_sobely_binary = np.zeros_like(gray_sobely_scaled)
    # Apply threshold and generate binary image
    gray_sobely_binary[(gray_sobely_scaled >= sx_thresh[0]) & (gray_sobely_scaled <= sx_thresh[1])] = 1

    # Threshold S channel
    hls_s_binary = np.zeros_like(hls_s)
    hls_s_binary[(hls_s >= s_thresh[0]) & (hls_s <= s_thresh[1])] = 1
    hls_s_binary = np.uint8(hls_s_binary)
    hls_s_binary_show = hls_s_binary * 255

    # Direction binary
    dir_binary = dir_threshold(img_rgb, sobel_kernel=3, thresh=(0.7, 1.3))
    dir_binary_out = dir_binary * 255

    # Output binary
    out = gray_sobelx_binary + luv_l_binary
    out_binary =  np.zeros_like(out)
    out_binary[(out > 0) & (out <= 2)] = 1
    out_binary_show = out_binary*255

    # Stack s_binary and gray_sobelx_binary
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    dummy = np.zeros_like(gray_sobelx_binary)
    color_stack = np.dstack((gray_sobelx_binary, dummy, hls_s_binary))*255
    color_stack_gray = cv2.cvtColor(color_stack, cv2.COLOR_RGB2GRAY)
    color_stack_binary = np.zeros_like(gray_sobelx_scaled)
    color_stack_binary[(color_stack_gray >= sx_thresh[0]) & (color_stack_gray <= sx_thresh[1])] = 1
    color_stack_binary_show = color_stack_binary * 255

    output_binary = out_binary
    output_binary_show = output_binary * 255

    imshape = output_binary.shape
    vertices = np.array(
        [[(0 + 50, imshape[0]), (imshape[1] / 2, imshape[0] / 2 + 30), (imshape[1] / 2, imshape[0] / 2 + 30),
          (imshape[1] - 50, imshape[0])]], dtype=np.int32)

    # This is the final output: luv_l_binary + gray_sobelx_binary
    output_binary_masked = region_of_interest(output_binary, vertices)

    # This is just to experiment with hls_s and sobelx
    color_stack_binary_masked = region_of_interest(color_stack_binary, vertices)

    if show_results == "YES":
        f, ax = plt.subplots(3, 3)
        plt.title("Threshold Images")
        bx = np.reshape(ax, 9)
        bx[0].imshow(img_rgb)
        bx[0].set_title('1. input')

        bx[1].imshow(luv_l, cmap='gray')
        bx[1].set_title('2. luv_l')

        bx[2].imshow(hls_s,cmap='gray')
        bx[2].set_title('3. hls_s')

        bx[3].imshow(gray_sobelx_binary_show,cmap='gray')
        bx[3].set_title('4. gray_sobelx_binary')

        bx[4].imshow(luv_l_binary_show, cmap='gray')
        bx[4].set_title('5. luv_l_binary')

        bx[5].imshow(hls_s_binary_show,cmap='gray')
        bx[5].set_title('6. hls_s_binary')

        bx[6].imshow(color_stack_binary_show, cmap='gray')
        bx[6].set_title('7.hls_s_binary + gray_sobelx_binary')

        bx[7].imshow(output_binary_show, cmap='gray')
        bx[7].set_title('8. luv_l_binary + gray_sobelx_binary')

        bx[8].imshow(output_binary_masked, cmap='gray')
        bx[8].set_title('9. OUTPUT: #8 masked')

        bx[0].axis('off')
        bx[1].axis('off')
        bx[2].axis('off')
        bx[3].axis('off')
        bx[4].axis('off')
        bx[5].axis('off')
        bx[6].axis('off')
        bx[7].axis('off')
        bx[8].axis('off')
        plt.show()

    return output_binary_masked

if threshold_show_results == "YES":
    for fname in testImageFileNames:
        img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
        threshold(img,"YES")

#===================================================================
# Step 4. Implement Perspective transform that warps the image
# to creates parallel lines
#====================================================================
# Read in an image
img_in = cv2.cvtColor( cv2.imread('test_images/straight_lines1.jpg'), cv2.COLOR_BGR2RGB)
img = img_in.copy()
# Manually select source points and define the destination points.
# Experiment a little bit with lines to come up with the best source region.
coef=80
s1 = [595,450]
s2 = [684,450]
s3 = [1108,720]
s4 = [202,s3[1]]
d1 = [s4[0]+coef,0]
d2 = [s3[0]-coef,0]
d3 = [s3[0]-coef,s3[1]]
d4 = [s4[0]+coef,s4[1]]
img = cv2.line(img,(s1[0],s1[1]),(s2[0],s2[1]),(255,0,0),5)
img = cv2.line(img,(s2[0],s2[1]),(s3[0],s3[1]),(255,0,0),5)
img = cv2.line(img,(s3[0],s3[1]),(s4[0],s4[1]),(255,0,0),5)
img = cv2.line(img,(s4[0],s4[1]),(s1[0],s1[1]),(255,0,0),5)
img_size = (img.shape[1], img.shape[0])
src = np.float32([s1, s2, s3, s4])
dst = np.float32([d1, d2, d3, d4])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

#===================================================================
# Step 5. Implement Warp function
#
#====================================================================
def warp( img, show_results="NO" ):
    img_size = (img.shape[1], img.shape[0])
    img_warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    if show_results == "YES":
        img_warped = warp(img_in)
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img)
        ax1.set_title('Original Image')
        ax1.axis('off')
        ax2.imshow(img_warped)
        ax2.set_title('Warped Image')
        ax2.axis('off')
        plt.show()
    return img_warped

# Show test images
if warp_show_results == "YES":
    img_warped = warp( img_in, "YES" )

# The perspective-transformed images are shown in the figure below
if warp_show_results == "YES":
    ax = []
    f, ax = plt.subplots(3, 2)
    bx = np.reshape(ax, 6)
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(testImageFileNames):
        img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
        img_warped = warp( img )
        bx[idx].imshow(img_warped)
        bx[idx].set_title(fname, fontsize=10)
        bx[idx].axis('off')
    plt.show()

#===================================================================
# Image pre-processing\
#   1. Correct distortion
#   2. Apply gradient or color based threshold operation to
#       produce a binary image with detectable lines.
#   3. Apply perspective transform to warp the image to create
#       birds aye view and make lines look parallel
#====================================================================
def image_preprocess( img ):
    # Correct distortion
    img_undist = undistort(img)
    # Apply thresholds
    img_thres = threshold(img_undist)
    # Warp the image
    img_warp = warp(img_thres)
    return img_warp

if image_preprocess_show_results == "YES":
    ax = []
    f, ax = plt.subplots(3, 2)
    bx = np.reshape(ax, 6)
    for idx, fname in enumerate(testImageFileNames):
        img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
        img = image_preprocess( img )
        bx[idx].imshow( img, cmap='gray')
        bx[idx].set_title(fname)
        bx[idx].axis('off')
    plt.show()

#===================================================================
# Sliding window search
#   This function finds centroids
#====================================================================
# window settings
window_width = 50
window_height = 80  # Break image into 9 vertical layers since image height is 720
margin = 100  # How much to slide left and right for searching
max_dev = 100    # mac deviation compared the previous line location

def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output

def find_window_centroids(image, window_width=50, window_height=80, margin=100, l_center_prev=0, r_center_prev=0):
    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    r_sum = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(image.shape[1] / 2)

    if np.abs(l_center - l_center_prev) > margin:
        l_center = l_center_prev

    if np.abs(r_center - r_center_prev) > margin:
        r_center = r_center_prev

    line_distance = r_center - l_center
    l_diff = 0
    r_diff = 0
    l_detected = 1
    r_detected = 1
    count =0

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(image.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            image[int(image.shape[0] - (level + 1) * window_height):int(image.shape[0] - level * window_height), :],
            axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, image.shape[1]))
        l_center_new = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, image.shape[1]))
        r_center_new = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset

        #if (line_distance_new > line_distance-margin) & (line_distance_new < line_distance+margin):

        # If the new detection is withing the expected range accept it, otherwise
        # reject it and used predicted value instead
        l_detected = 0
        r_detected = 0

        # left line
        if np.abs(l_center_new - l_center) < max_dev:
            l_detected = 1

        # right line
        if np.abs(r_center_new - r_center) < max_dev:
            r_detected = 1

        # If both left and right lines were detected update the centers
        if ((l_detected == 1) & (r_detected == 1)):
            l_diff = l_center_new - l_center
            r_diff = r_center_new - r_center
            l_center = l_center_new
            r_center = r_center_new

        # If the right line was detected but the left line was not detected, calculate the predicted
        # location of the left line based on the current position of the right line and the expected distance
        # between the lines.
        if ((l_detected == 0) & (r_detected == 1)):
            r_diff = r_center_new - r_center
            r_center = r_center_new
            l_center = r_center - line_distance
            l_diff = r_diff

        # If the left line was detected but the right line was not detected, calculate the predicted
        # location of the right line based on the current position of the right line and the expected distance
        # between the lines.
        if ((l_detected == 1) & (r_detected == 0)):
            l_diff = l_center_new - l_center
            l_center = l_center_new
            r_center = l_center + line_distance
            r_diff = l_diff

        # If neither line was detected, predict the current location of the lines based on the previous data.
        if ((l_detected == 0) & (r_detected == 0)):
            diff = (l_diff + r_diff)/2
            l_center = l_center + diff
            r_center = r_center + diff

        #l_center = min(l_center, image.shape[1]-1)
        #l_center = max(l_center, 1)

        #r_center = min(r_center, image.shape[1] - 1)
        #r_center = max(r_center, 1)

        line_distance = r_center - l_center

        # Add what we found for that layer
        window_centroids.append((l_center, r_center))
        count += 1

    return window_centroids

#===================================================================
# Line parameters calculation
#====================================================================
def calc_line_parameters( image, window_centroids ):
    num_levels = (int)(image.shape[0] / window_height)
    leftx = np.linspace(0, num_levels - 1, num=num_levels)
    rightx = np.linspace(0, num_levels - 1, num=num_levels)
    left_fit = []
    right_fit = []
    ploty = np.linspace(0, num_levels - 1, num=num_levels)
    ploty = ploty *1.0
    for level in range(0, num_levels):
        ploty[level] = (level * window_height) + (window_height / 2)
    # If we found any window centers
    if len(window_centroids) > 0:
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(image)
        r_points = np.zeros_like(image)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, image, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, image, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
        zero_channel = np.zeros_like(template)  # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
        warpage = np.dstack((image, image, image)) * 255  # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results

        # Prepare data to fit the polynomial
        for level in range(0, len(window_centroids)):
            leftx[level] = window_centroids[level][0]
            rightx[level] = window_centroids[level][1]

        leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
        rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

        # Fit a second order polynomial to pixel positions for left and right lines
        left_fit = np.polyfit(ploty, leftx, 2)
        right_fit = np.polyfit(ploty, rightx, 2)

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((image, image, image)), np.uint8)

    return output, left_fit, right_fit, leftx, rightx, ploty
#===================================================================
# Line curvature and car position calculation
#
#====================================================================
def calc_curvature_pos( image, left_fit, right_fit, leftx, rightx, ploty ):
    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radius of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
    # Calculate vehicle center
    max_x = image.shape[1] * xm_per_pix
    max_y = image.shape[0] * ym_per_pix
    center = max_x / 2
    left_lane = left_fit_cr[0] * max_y ** 2 + left_fit_cr[1] * max_y + left_fit_cr[2]
    right_lane = right_fit_cr[0] * max_y ** 2 + right_fit_cr[1] * max_y + right_fit_cr[2]
    middle_lane = left_lane + (right_lane - left_lane) / 2
    pos = middle_lane - center
    left_pos = center - left_lane
    right_pos = right_lane - center

    return left_curverad, right_curverad, pos, left_pos, right_pos

def update_line_params(line, center, x, ploty, fit, curverad, pos, ploty_axis, fitx):
    if line:
        line.detected = 1
        line.center = center                                # line location in pixels
        line.recent_xfitted.append(x)                       # add new element (coeffs)
        line.n_curr += 1
        if line.n_curr > line.n_max:
            line.recent_xfitted.pop(0)                      # remove the last element from the list
            line.n_curr -= 1
        line.bestx = np.mean(line.recent_xfitted, axis=0)   # calculate average fitted line pixel values
        line.best_fit = np.polyfit(ploty, line.bestx, 2)    # calculate coefficients for the average line
        line.diffs = line.current_fit - fit                 # coefficient diffs
        line.current_fit = fit                              # coefficients
        line.radius_of_curvature = curverad                 # radius of curvature
        line.line_base_pos = pos                            # distance between left line and center of the car [m]
        line.allx = ploty_axis
        line.ally = fitx

#===================================================================
# Define the image processing pipeline
#
#===================================================================
def image_processing_pipeline( image, show_results="NO", l_center_prev=0, r_center_prev=0, l_line=None, r_line=None):

    ploty_axis = np.linspace(0, image.shape[0] - 1, num=image.shape[0])

    # Correct distortion
    image_undist = undistort(image)

    # Apply thresholds and generate binary image
    image_thres = threshold(image_undist)

    # Warp the binary image
    binary_warped = warp(image_thres)

    # Calculate Centroids
    window_centroids = find_window_centroids(binary_warped, window_width, window_height, margin, l_center_prev, r_center_prev)
    l_center = window_centroids[0][0]
    r_center = window_centroids[0][1]

    # Calculate line parameters
    img_centroid_fitting, left_fit, right_fit, leftx, rightx, ploty = calc_line_parameters(binary_warped, window_centroids)

    # Calculate curvature
    l_curverad, r_curverad, pos, left_pos, right_pos = calc_curvature_pos(image, left_fit, right_fit, leftx, rightx, ploty)
    if pos < 0:
        display = '{:.2f} m right'.format(pos)
    else:
        display = '{:.2f} m left'.format(-pos)

    # Plot the lines
    left_fitx = left_fit[0] * ploty_axis ** 2 + left_fit[1] * ploty_axis + left_fit[2]
    right_fitx = right_fit[0] * ploty_axis ** 2 + right_fit[1] * ploty_axis + right_fit[2]

    # Update left and right line parameters
    update_line_params(l_line, l_center, leftx, ploty, left_fit, l_curverad, left_pos, ploty_axis, left_fitx)
    update_line_params(r_line, r_center, rightx, ploty, right_fit, r_curverad, right_pos, ploty_axis, right_fitx)

    # Use the best fit instead of the current fit parameters for the left and right lines
    if l_line:
        left_fit = l_line.best_fit
        left_fitx = left_fit[0] * ploty_axis ** 2 + left_fit[1] * ploty_axis + left_fit[2]
    if r_line:
        right_fit= r_line.best_fit
        right_fitx = right_fit[0] * ploty_axis ** 2 + right_fit[1] * ploty_axis + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty_axis]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty_axis])))])
    pts = np.hstack((pts_left, pts_right))


    # Draw the lane onto the centroid image
    # Python: cv.PolyLine(img, polys, is_closed, color, thickness=1, lineType=8, shift=0) → None
    img_lines_fitting = cv2.polylines(img_centroid_fitting, np.int_([pts]), True, (0, 0, 255), 10, 2)
    if image_processing_pipeline_show_line_filling == "YES":
        plt.imshow(img_lines_fitting)
        plt.show()

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(image_undist, 1, newwarp, 0.3, 0)

    # Draw info
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(result, 'Left Curvature: {:.0f} m'.format(l_curverad), (50, 50), font, 2, (0, 255, 0), 2)
    cv2.putText(result, 'Right Curvature: {:.0f} m'.format(r_curverad), (50, 100), font, 2, (0, 255, 0), 2)
    cv2.putText(result, 'Vehicle is {} of center'.format(display), (50, 150), font, 2, (0, 255, 0), 2)

    if show_results == "YES":
        # Display the image with lines fitting
        #plt.imshow(img_lines_fitting)
        #plt.plot(left_fitx, ploty_axis, color='green', linewidth=3)
        #plt.plot(right_fitx, ploty_axis, color='green', linewidth=3)
        #plt.title('window fitting results')
        ax = []
        f, ax = plt.subplots(4, 2)
        bx = np.reshape(ax, 8)
        bx[0].imshow(image)
        bx[0].set_title("Input Image")
        bx[0].axis('off')
        bx[1].imshow(image_undist)
        bx[1].set_title("Undistorted Image")
        bx[1].axis('off')
        bx[2].imshow(image_thres)
        bx[2].set_title("Threshold Image")
        bx[2].axis('off')
        bx[3].imshow(binary_warped)
        bx[3].set_title("Warped Image")
        bx[3].axis('off')
        bx[4].imshow(img_lines_fitting)
        bx[4].set_title("Lines fitting")
        bx[4].axis('off')
        bx[5].imshow(color_warp)
        bx[5].set_title("Color warp")
        bx[5].axis('off')
        bx[6].imshow(newwarp)
        bx[6].set_title("Color Unwarp")
        bx[6].axis('off')
        bx[7].imshow(result)
        bx[7].set_title("Output Image")
        bx[7].axis('off')

        plt.show()

    return result, l_center, r_center

#===================================================================
# Process test images
#===================================================================
if process_test_images == "YES":
    img_cnt = 0
    show_results = process_test_images_show_results
    l_center = 310
    r_center = 1070
    img_count = testImageFileNames
    rows = 3
    cols = 2
    ax = []
    f,ax = plt.subplots(rows, cols)
    bx = np.reshape(ax, rows*cols)
    for idx, fname in enumerate(testImageFileNames):
        #if img_cnt > 80:
        #   show_results = "YES"
        #fname = "test/" + "img_in" + np.str(img_cnt) + ".jpg"
        img_in = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
        img_out, l_center, r_center = image_processing_pipeline(img_in,show_results,l_center,r_center,None,None)
        bx[idx].imshow(img_out)
        bx[idx].set_title(fname)
        bx[idx].axis('off')
        img_cnt += 1
    plt.show()

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # line location in pixels
        self.center = None
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
        # Max number of elements in the history buffer
        self.n_max = 5
        # Current number of elements in the history buffer
        self.n_curr = 0

class videoPipeline():
    def __init__(self):
        self.img_cnt = 0
        self.l_line = Line()
        self.l_line.center = 310.0
        self.r_line = Line()
        self.r_line.center = 1076.0

    def image_processing_pipeline(self, img_in):

        show_results = process_video_show_results

        #if self.img_cnt > 80:
        #    show_results = "YES"
        if process_video_dump_imgs == "YES":
            fname = "test/" + "img_in" + np.str(self.img_cnt) + ".jpg"
            image_bgr = cv2.cvtColor(img_in, cv2.COLOR_RGB2BGR)
            cv2.imwrite(fname,image_bgr)

        img_out, l_center, r_center = image_processing_pipeline(
            img_in, show_results, self.l_line.center, self.r_line.center, self.l_line, self.r_line )

        self.l_line.center = l_center
        self.r_line.center = r_center

        if process_video_dump_imgs == "YES":
            fname = "test/" + "img_out" + np.str(self.img_cnt) + ".jpg"
            img_out_bgr = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
            cv2.imwrite(fname, img_out_bgr)

        self.img_cnt += 1
        return img_out

#===================================================================
# Video processing pipeline
#===================================================================
from moviepy.editor import VideoFileClip
def video_pipeline( input_file, output_file ):

    vp = videoPipeline()

    #in_clip = VideoFileClip(input_file).subclip(38,43)
    in_clip = VideoFileClip(input_file)

    out_clip = in_clip.fl_image(vp.image_processing_pipeline)

    out_clip.write_videofile(output_file, audio=False)

#===================================================================
# Process the video
#===================================================================
if process_video == "YES":
    video_pipeline('project_video.mp4', 'project_video_out.mp4')

