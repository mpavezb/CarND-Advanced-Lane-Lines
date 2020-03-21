import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.logger import Log
from src.calibration import CalibrationParameters, CameraModel


def RunCalibration(display=False):
    Log.section("Camera Calibration")

    params = CalibrationParameters()
    camera = CameraModel(params)
    camera.calibrate()

    if display:
        camera.display_calibration()

    # getting the points
    objpoints, imgpoints = camera.get_3d_to_2d_points()

    Log.success()
    return camera


def RunDistortionCorrectionExample(camera):

    # Make a list of test images
    #   Here I selected some chessboard images, were the distortion
    #   correction result is notorious.
    images = [
        "camera_cal/calibration1.jpg",
        "camera_cal/calibration2.jpg",
        "camera_cal/calibration3.jpg",
        "camera_cal/calibration4.jpg",
    ]

    Log.subsection("Correct Distortion on sample images")

    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(10, 10))
    fig.tight_layout()
    axs[0, 0].set_title("Original Image")
    axs[0, 1].set_title("Undistorted Image")
    for idx, fname in enumerate(images):
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
        axs[idx, 0].imshow(img)
        axs[idx, 1].imshow(undistorted)
        axs[idx, 0].axis("off")
        axs[idx, 1].axis("off")

    plt.show()
    Log.success()


def main():
    Log.debug_enabled = False
    camera = RunCalibration(display=True)
    # RunDistortionCorrectionExample(camera)

    # wait for user to finish program
    plt.waitforbuttonpress()


if __name__ == "__main__":
    main()
