import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.logger import Log
from src.calibration import CalibrationParameters, CameraModel


def RunCalibration(display=True):
    Log.section("Camera Calibration")

    params = CalibrationParameters()
    params.display = display

    camera = CameraModel(params)
    camera.calibrate()

    # getting the points
    objpoints, imgpoints = camera.get_3d_to_2d_points()

    Log.success()
    return camera


def RunDistortionCorrectionExample(camera):

    # Make a list of test images
    #   Here I selected some chessboard images, were the distortion
    #   correction is notorious.
    images = [
        "camera_cal/calibration1.jpg",
        "camera_cal/calibration2.jpg",
        "camera_cal/calibration3.jpg",
        "camera_cal/calibration4.jpg",
    ]

    print("> Correct Distortion on sample images:")
    fig = plt.figure(figsize=(24, 9))
    # ax1.set_title("Original Image", fontsize=50)
    # ax2.set_title("Undistorted Image", fontsize=50)
    fig.tight_layout()
    fig_cnt = 0
    for fname in images:
        Log.info("file: " + fname)

        # Read grayscale
        img_ = cv2.imread(fname)
        img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)

        # Apply distortion correction
        w = img.shape[1]
        h = img.shape[0]
        [mtx, dist] = camera.get_calibration(w, h)
        undistorted = cv2.undistort(img, mtx, dist, None, mtx)

        # Display Comparison
        fig_cnt = fig_cnt + 1
        fig.add_subplot(4, 2, 2 * fig_cnt - 1)
        plt.imshow(img)

        fig.add_subplot(4, 2, 2 * fig_cnt)
        plt.imshow(undistorted)

    Log.success()


def main():
    Log.debug_enabled = False
    camera = RunCalibration(display=False)
    RunDistortionCorrectionExample(camera)

    # wait for user to finish program
    plt.waitforbuttonpress()


if __name__ == "__main__":
    main()
