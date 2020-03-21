import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

from .logger import Log
from .cache import Cache


class CalibrationParameters:
    glob = "camera_cal/calibration*.jpg"
    pickle = "calibration.p"
    nx = 9
    ny = 6


class CameraModel:
    """Computes objpoints,imgpoints pair based on chessboard images for calibration"""

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space.
    imgpoints = []  # 2d points in image plane.
    images = []  # images from which these points where computed.

    # Calibration
    cal_w = None
    cal_h = None
    mtx = None
    dist = None

    def __init__(self, parameters):
        self.nx = parameters.nx
        self.ny = parameters.ny
        self.target_images = glob.glob(parameters.glob)

        # cache
        self.cache = Cache(parameters.pickle)

    def save(self):
        data = dict()
        data["objpoints"] = self.objpoints
        data["imgpoints"] = self.imgpoints
        data["images"] = self.images
        self.cache.save(data)

    def load(self):
        if self.cache.exists():
            data = self.cache.load()
            self.objpoints = data["objpoints"]
            self.imgpoints = data["imgpoints"]
            self.images = data["images"]
            return True
        return False

    def calibrate(self):
        if self.load():
            Log.subsection("Using cached calibration data ...")
            return
        Log.subsection("Running calibration on %d images ..." % len(self.target_images))

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.nx * self.ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0 : self.nx, 0 : self.ny].T.reshape(-1, 2)

        # Step through the list and search for chessboard corners
        for fname in self.target_images:
            self.calibrate_single(fname, objp)

        # Update cache
        self.save()

    def calibrate_single(self, fname, objp):
        Log.info("file: " + fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)

        # If found, save object points, image points
        if ret == True:
            self.images.append(fname)
            self.objpoints.append(objp)
            self.imgpoints.append(corners)
        else:
            Log.warn(
                "cv2.findChessboardCorners was not able to process file: %s" % fname
            )

    def display_calibration(self):
        fig = plt.figure(1, figsize=(20, 15))

        # Draw and display the corners
        for idx, fname in enumerate(self.images):
            # draw on image
            img = cv2.imread(fname)
            chessboard = cv2.drawChessboardCorners(
                img, (self.nx, self.ny), self.imgpoints[idx], True
            )

            # plot
            fig.add_subplot(5, 4, idx + 1)
            plt.imshow(chessboard)
            plt.axis("off")

    def get_3d_to_2d_points(self):
        return self.objpoints, self.imgpoints

    def get_calibration(self, w, h):
        # Use cached
        if w is self.cal_w and h is self.cal_h:
            Log.debug("Using Cached")
            return self.mtx, self.dist
        Log.debug("Computing")

        # Compute
        [_, self.mtx, self.dist, _, _] = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, (w, h), None, None
        )

        self.cal_w = w
        self.cal_h = h
        return self.mtx, self.dist
