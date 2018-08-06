## Writeup

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

[//]: # (Image References)

[video1]: ./output1_tracked.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in `cameracaliberation.py`

Camera is prone to create distortions while capturing images from the real word and in this step followed the lecture and corrected the distortions of th test images by caliberating the camera using chessboard images. The "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. The chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time the chessboard corners is successfuly detected in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

![input image](camera_cal/calibration1.jpg)  | ![output image](output_images/draw_chess_board/calibration6.jpg)
---------------------------------------------| -----------------------------------------------------------------

The output `objpoints` and `imgpoints` are used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function refer `undistortimages.py`.  The distortion correction is applied to the test image using the `cv2.undistort()` function to obtain this result: 

![input image](camera_cal/calibration1.jpg)  | ![output image](output_images/chessundistorted/calibration1.jpg)
---------------------------------------------| -----------------------------------------------------------------


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Below is one of the distortion corrected test image where the white vehicle is displayed correctly after distiortion correction refer `undistortimages.py` for opencv code used to correct distortion:

![input image](test_images/test1.jpg)  | ![output image](output_images/undistorted/test1.jpg)
-------------------------------------- | -----------------------------------------------------------------



#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

As suggested in the lecture video a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # 46 # 87 `combinethresholds.py`). The 'S' and 'V' color channels are used to make sure the lanes are still present after this step and also gradient absolute theshold is applied as well and following is the output after this step:

![input image](test_images/test1.jpg)  | ![output image](output_images/combinethresholds/test1.jpg)
-------------------------------------- | -----------------------------------------------------------------
![input image](test_images/test4.jpg)  | ![output image](output_images/combinethresholds/test4.jpg)

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for perspective transform is included in a function called `pipeline()`, which appears in lines 64 through 79 in the file `laneimageprocessor.py`.  The `pipeline` function takes as inputs an image (`img`) after combine threshold image processing step and hard coded src and dst points is choosen as below for the transformation

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

The perspective transform was verfied working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![input image](test_images/test1.jpg)  | ![output image](output_images/bird_eye/test1.jpg)
-------------------------------------- | -----------------------------------------------------------------
![input image](test_images/test4.jpg)  | ![output image](output_images/binary_lanes/test4.jpg)


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![input image](test_images/test1.jpg)  | ![output image](output_images/tracker/test1.jpg)
-------------------------------------- | -----------------------------------------------------------------

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![input image](test_images/test2.jpg)  | ![output image](output_images/draw_lane/test2.jpg)
-------------------------------------- | -----------------------------------------------------------------

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
