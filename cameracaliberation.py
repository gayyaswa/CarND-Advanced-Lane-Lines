
import  numpy  as  np
import  cv2
import glob
import os
import pickle

chess_draw_dir_name = 'output_images/draw_chess_board'

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((6*9,3), np.float32)

objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
#print(objp)

images = glob.glob('camera_cal/calibration*.jpg')

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points
    if ret is True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img,(9,6),corners,ret)

        #print(fname)
        os.makedirs(os.path.dirname(os.path.join(chess_draw_dir_name, '')), exist_ok=True)
        full_name = os.path.join(chess_draw_dir_name, os.path.basename(fname))
        #print(full_name)
        cv2.imwrite(full_name, img)

#Caliberate the camera image
img = cv2.imread('camera_cal/calibration2.jpg')
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)

#save camera caliberation
dist_pickle = {}
dist_pickle['mtx'] = mtx
dist_pickle['dist'] = dist
pickle.dump( dist_pickle, open('caliberation_dist_pickle.p','wb'))
