import numpy as np
import cv2
import glob
import os
from tracker import Tracker
from combinethresholds import CombineThreshold


# Read in an image
out_dir_name_bin_lane = 'output_images/binary_lanes/'
out_dir_name_bird_eye = 'output_images/bird_eye/'
out_dir_name_draw_lane = 'output_images/draw_lane/'
out_dir_name_lane_tracker = 'output_images/tracker/'


def generate_bin_lane_warp(image, binary_warped, Minv):
    #images = glob.glob('output_images/binary_lanes/*.jpg')
    # print(images)
    #for fname in images:
        #binary_warped = mpimg.imread(fname)
    genpolyline = Tracker()
    lane_on_warped_blank_img, left_fitx, right_fitx, ploty = genpolyline.search_around_poly(binary_warped)
    os.makedirs(os.path.dirname(os.path.join(out_dir_name_lane_tracker, '')), exist_ok=True)
    full_name = os.path.join(out_dir_name_lane_tracker, os.path.basename(fname))
    # print(full_name)
    cv2.imwrite(full_name, lane_on_warped_blank_img)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    #plt.imshow(result)

    os.makedirs(os.path.dirname(os.path.join(out_dir_name_draw_lane, '')), exist_ok=True)
    full_name = os.path.join(out_dir_name_draw_lane, os.path.basename(fname))
    cv2.imwrite(full_name, result)





# Run the function
# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
#mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(0, 255))
#dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi/2))


combinethreshold = CombineThreshold()

images = glob.glob('output_images/undistorted/*.jpg')
#print(images)
for fname in images:
    img = cv2.imread(fname)
    line_bin_image = np.zeros_like(img[:,:,0])
    gradx = combinethreshold.abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(12, 255))
    grady = combinethreshold.abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(25, 255))
    c_binary = combinethreshold.color_thresh(img, s_thresh=(100,255), v_thresh=(50,255))

    line_bin_image[((gradx == 1) & (grady == 1)) | (c_binary == 1)] = 255

    img_height, img_width, channels = img.shape
    img_size = (img.shape[1], img.shape[0])
    print(img_size)

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

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    Minv = cv2.getPerspectiveTransform(dst, src)

    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(line_bin_image, M, (img_width,img_height), flags=cv2.INTER_LINEAR)
    warped_on_undist = cv2.warpPerspective(img, M, (img_width, img_height), flags=cv2.INTER_LINEAR)


    os.makedirs(os.path.dirname(os.path.join(out_dir_name_bin_lane, '')), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.join(out_dir_name_bird_eye, '')), exist_ok=True)
    full_name = os.path.join(out_dir_name_bin_lane, os.path.basename(fname))
    print(full_name)
    cv2.imwrite(full_name, warped)
    full_name = os.path.join(out_dir_name_bird_eye, os.path.basename(fname))
    print(full_name)
    cv2.imwrite(full_name, warped_on_undist)

    generate_bin_lane_warp(img, warped, Minv)

# Plot the result
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# f.tight_layout()
# ax1.imshow(image)
# ax1.set_title('Original Image', fontsize=50)
# ax2.imshow(dir_binary, cmap='gray')
# ax2.set_title('Thresholded Grad. Dir.', fontsize=50)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)