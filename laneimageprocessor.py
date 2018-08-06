import cv2
import pickle
import numpy as np
import glob

from tracker import Tracker
from combinethresholds import CombineThreshold
from moviepy.editor import VideoFileClip

out_dir_name_draw_lane = 'output_images/draw_lane/'

class LaneImageProcessor:

    def __init__(self):
        #load the calibration coefficients once
        dist_pickle = pickle.load(open("caliberation_dist_pickle.p", "rb"))
        self.mtx = dist_pickle["mtx"]
        self.dist = dist_pickle["dist"]
        self.tracker = Tracker(False)
        self.combinethreshold = CombineThreshold()
        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        return

    def measure_curvature_real(self, left_fit, right_fit, ploty):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters


        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)

        left_fit_cr = np.polyfit(ploty * self.ym_per_pix, left_fit * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * self.ym_per_pix, right_fit * self.xm_per_pix, 2)


        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * self.ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * self.ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        return left_curverad, right_curverad


    def pipeline(self, img):

        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

        line_bin_image = np.zeros_like(img[:, :, 0])
        ksize = 3
        gradx = self.combinethreshold.abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(12, 255))
        grady = self.combinethreshold.abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(25, 255))
        c_binary = self.combinethreshold.color_thresh(img, s_thresh=(100, 255), v_thresh=(50, 255))

        line_bin_image[((gradx == 1) & (grady == 1)) | (c_binary == 1)] = 255

        img_height, img_width, channels = img.shape
        img_size = (img.shape[1], img.shape[0])
        #print(img_size)

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
        binary_warped = cv2.warpPerspective(line_bin_image, M, (img_width, img_height), flags=cv2.INTER_LINEAR)

        lane_on_warped_blank_img, left_fitx, right_fitx, ploty = self.tracker.search_around_poly(binary_warped)

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
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

        left_curverad, right_curverad = self.measure_curvature_real(left_fitx, right_fitx, ploty)

        radius_of_curvature = (left_curverad + right_curverad)/2
        camera_center = (left_fitx[-1] + right_fitx[-1])/2
        center_diff = (camera_center - newwarp.shape[1]/2) * self.xm_per_pix

        side_pos = 'left'
        if center_diff <=0:
            side_pos = 'right'

        #add radius and offset to the result
        #For straight road larger radius are being caculated so added a threhold and simply output "Straight Road"
        # for those cases
        # if radius_of_curvature > 1500:
        #     cv2.putText(result, 'Straight Road ahead', (50, 50),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # else:
        #     cv2.putText(result, 'Radius of curvature = ' + str(round(radius_of_curvature, 3)) + ' (m)', (50, 50),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, 'Radius of curvature = ' + str(round(radius_of_curvature, 3)) + ' (m)', (50, 50),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, 'Vehicle is ' + str(abs(round(center_diff, 3))) + 'm ' + side_pos + ' of center',
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return result


if __name__ == "__main__":
    images = glob.glob('output_images/undistorted/*.jpg')
    # print(images)
    laneimageprocessor = LaneImageProcessor()
    # for fname in images:
    #     img = cv2.imread(fname)
    #     result = laneimageprocessor.pipeline(img)
    #     os.makedirs(os.path.dirname(os.path.join(out_dir_name_draw_lane, '')), exist_ok=True)
    #     full_name = os.path.join(out_dir_name_draw_lane, os.path.basename(fname))
    #     cv2.imwrite(full_name, result)


    output_video = 'output1_tracked.mp4'
    input_video = 'project_video.mp4'

    clip1 = VideoFileClip(input_video)
    video_clip = clip1.fl_image(laneimageprocessor.pipeline)
    video_clip.write_videofile(output_video, audio=False)


