import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

from .logger import Log


class CalibrationParameters:
    glob = "camera_cal/calibration*.jpg"
    nx = 9
    ny = 6
    display = False


class CameraModel:
    """Computes objpoints,imgpoints pair based on chessboard images for calibration"""

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    def __init__(self, parameters):
        self.nx = parameters.nx
        self.ny = parameters.ny
        self.images = glob.glob(parameters.glob)
        self.display = parameters.display

    def calibrate(self):
        Log.subsection("Running calibration on %d images ..." % len(self.images))

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.nx * self.ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0 : self.nx, 0 : self.ny].T.reshape(-1, 2)

        self.display_setup()

        # Step through the list and search for chessboard corners
        for fname in self.images:
            Log.info("file: " + fname)
            img = cv2.imread(fname)

            if not self.calibrate_single(img, objp):
                Log.warn(
                    "cv2.findChessboardCorners was not able to process file: %s" % fname
                )
        Log.success()

    def calibrate_single(self, img, objp):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)

        # If found, save object points, image points
        if ret == True:
            self.objpoints.append(objp)
            self.imgpoints.append(corners)

            # Draw and display the corners
            self.display_update(img, corners, ret)
        return ret

    def display_setup(self):
        if self.display:
            self.fig = plt.figure(1, figsize=(20, 15))
            self.plot_number = 0

    def display_update(self, img, corners, ret):
        if self.display:
            self.plot_number = self.plot_number + 1
            self.fig.add_subplot(5, 4, self.plot_number)
            img = cv2.drawChessboardCorners(img, (self.nx, self.ny), corners, ret)
            plt.imshow(img)

    def get_calibration(self):
        return self.objpoints, self.imgpoints
