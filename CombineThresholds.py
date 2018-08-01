import numpy as np
import cv2
import glob
import os

# Read in an image
out_dir_name = 'output_images/binary_lanes'


# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
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
    orient_img = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    sxbinary = np.zeros_like(orient_img)
    sxbinary[(orient_img >= thresh[0]) & (orient_img <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return sxbinary

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
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
def abs_sobel_thresh(img, orient='x', sobel_kernel=3,thresh=(0, 255)):
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

def color_thresh(img, s_thresh=(0, 255), v_thresh=(0, 255)):
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


# Run the function
# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
#mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(0, 255))
#dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi/2))

images = glob.glob('output_images/undistorted/*.jpg')
#print(images)
for fname in images:
    img = cv2.imread(fname)
    line_bin_image = np.zeros_like(img[:,:,0])
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(12, 255))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(25, 255))
    c_binary = color_thresh(img, s_thresh=(100,255), v_thresh=(50,255))

    line_bin_image[((gradx == 1) & (grady == 1)) | (c_binary == 1)] = 255

    img_height, img_width, channels = img.shape

    #define trapezoid area of interest
    trap_bottom_width = .76
    trap_mid_width = .08
    trap_height = 0.62
    bottom_trim = .935
    src = np.float32([[img_width * (.5 - trap_mid_width/2), img_height * trap_height], [img_width *(.5 + trap_mid_width/2), img_height * trap_height ], ])
    os.makedirs(os.path.dirname(os.path.join(out_dir_name, '')), exist_ok=True)
    full_name = os.path.join(out_dir_name, os.path.basename(fname))
    #print(full_name)
    cv2.imwrite(full_name, line_bin_image)

# Plot the result
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(image)
# ax1.set_title('Original Image', fontsize=50)
# ax2.imshow(dir_binary, cmap='gray')
# ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)