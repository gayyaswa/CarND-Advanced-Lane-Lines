import cv2
import pickle
import glob
import os

undistorted_dir_name = 'output_images/undistorted'
chess_undistor_dir_name ="output_images/chessundistorted"

# Read in the saved objpoints and imgpoints
dist_pickle = pickle.load( open( "caliberation_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

def undistort(images, undistorted_dir_name):

    for fname in images:
        img = cv2.imread(fname)
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        os.makedirs(os.path.dirname(os.path.join(undistorted_dir_name, '')), exist_ok=True)
        full_name = os.path.join(undistorted_dir_name, os.path.basename(fname))
        #print(full_name)
        cv2.imwrite(full_name, undist)
    return

images = glob.glob('test_images/*.jpg')
undistort(images,undistorted_dir_name)

images = glob.glob('camera_cal/*.jpg')
undistort(images,chess_undistor_dir_name)
#print(images)
