import numpy as np
import cv2
import glob
import os
from tracker import Tracker


# Read in an image
out_dir_name_bin_lane = 'output_images/binary_lanes/'
out_dir_name_bird_eye = 'output_images/bird_eye/'
out_dir_name_draw_lane = 'output_images/draw_lane/'
out_dir_name_lane_tracker = 'output_images/tracker/'

class CombineThreshold:
    # Define a function that applies Sobel x and y,
    # then computes the direction of the gradient
    # and applies a threshold.
    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi / 2)):
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
        orient_img = np.arctan2(abs_sobely, abs_sobelx)
        # 5) Create a binary mask where direction thresholds are met
        sxbinary = np.zeros_like(orient_img)
        sxbinary[(orient_img >= thresh[0]) & (orient_img <= thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        return sxbinary

    # Define a function to return the magnitude of the gradient
    # for a given sobel kernel size and threshold values
    def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255
        gradmag = (gradmag/scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        # Return the binary image
        return binary_output

    # Define a function that takes an image, gradient orientation,
    # and threshold min / max values.
    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3,thresh=(0, 255)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        # Return the result
        return binary_output

    def color_thresh(self, img, s_thresh=(0, 255), v_thresh=(0, 255)):
        # 1) Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        # 2) Apply a threshold to the S channel
        s = hls[:, :, 2]
        # 3) Return a binary image of threshold result
        s_binary = np.zeros_like(s)
        s_binary[(s > s_thresh[0]) & (s <= s_thresh[1])] = 1

        # 1) Convert to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # 2) Apply a threshold to the v channel
        v = hsv[:, :, 2]
        # 3) Return a binary image of threshold result
        v_binary = np.zeros_like(v)
        v_binary[(v > v_thresh[0]) & (v <= v_thresh[1])] = 1

        output = np.zeros_like(s)
        output[(s_binary == 1) & (v_binary == 1)] = 1
        return output



