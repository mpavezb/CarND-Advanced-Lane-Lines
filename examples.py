import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.logger import Log
from src.calibration import GetCalibratedCamera
from src.filtering import EdgeDetector
from src.save import save_image


def RunCalibrationExample():
    Log.section("Camera Calibration")
    camera = GetCalibratedCamera()
    camera.display_calibration()


def RunDistortionCorrectionExample():
    Log.section("Distortion Correction")
    camera = GetCalibratedCamera()

    # Make a list of test images
    #   Here I selected some chessboard images, were the distortion
    #   correction result is notorious.
    images = [
        "camera_cal/calibration1.jpg",
        "camera_cal/calibration2.jpg",
        "camera_cal/calibration3.jpg",
        "camera_cal/calibration4.jpg",
    ]

    # Correct distortion save and display results
    Log.subsection("Correct Distortion on sample images")
    f, axs = plt.subplots(len(images), 1, figsize=(10, 10))
    f.tight_layout()
    axs[0].set_title("Original and Undistorted Images", fontsize=15)
    for idx, fname in enumerate(images):
        # Read grayscale
        Log.info("image: " + fname)
        img_ = cv2.imread(fname)
        img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)

        # Distortion Correction
        undistorted = camera.undistort(img)

        # Concatenate
        vis = np.concatenate((img, undistorted), axis=1)

        # Display Comparison
        axs[idx].imshow(vis)
        axs[idx].axis("off")

        # Save
        save_image(vis, fname, "distortion_correction_")

    out_fname = os.path.join("output_images", "distortion_correction.png")
    Log.info("Saving figure to %s" % out_fname)
    plt.savefig(out_fname)

    Log.subsection("Display")
    plt.show()
    Log.success()


def RunEdgeDetectionExample():
    Log.section("EdgeDetection")
    camera = GetCalibratedCamera()
    images = glob.glob("test_images/*.jpg")
    for idx, fname in enumerate(images):
        Log.subsection("Processing image: %s" % fname)
        image = cv2.imread(fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply distortion correction
        Log.info("Distortion correction ...")
        undistorted = camera.undistort(image)

        # Detect Edges
        Log.info("Edge Detection ...")
        edge_detector = EdgeDetector()
        binary = edge_detector.detect(image)

        # Display and Save Results
        Log.info("Display Results ...")
        plt.figure(idx)
        edge_detector.display_pipeline(fname)
    plt.show()


def warp(image):
    h = 720
    l = 220
    r = 1110
    t = 440
    tl = 610
    tr = 670
    src = np.float32([[l, h], [tl, t], [tr, t], [r, h]])
    dst = np.float32([[l, h], [l, 0], [r, 0], [r, h]])

    img_size = (image.shape[1], image.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, src, dst


def RunPerspectiveTransformExample():
    Log.section("Perspective Transform")
    camera = GetCalibratedCamera()

    # image set
    images = glob.glob("test_images/*.jpg")

    # Correct distortion save and display results
    Log.subsection("Apply warp to sample images")
    f, axs = plt.subplots(len(images), 1, figsize=(10, 10))
    f.tight_layout()
    axs[0].set_title("Undistorted and Warped Images", fontsize=15)
    for idx, fname in enumerate(images):
        # Read
        Log.info("image: " + fname)
        image = cv2.imread(fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Warp
        undistorted = camera.undistort(image)
        warped, src, dst = warp(undistorted)

        # Draw
        cv2.polylines(undistorted, [np.int32(src)], 1, (255, 0, 0), thickness=1)
        cv2.polylines(warped, [np.int32(dst)], 1, (255, 0, 0), thickness=1)

        # Concatenate
        vis = np.concatenate((undistorted, warped), axis=1)

        # Display Comparison
        axs[idx].imshow(vis)
        axs[idx].axis("off")

        # Save
        save_image(vis, fname, "warp_")

    out_fname = os.path.join("output_images", "warp.png")
    Log.subsection("Saving figure to %s" % out_fname)
    plt.savefig(out_fname)

    Log.subsection("Display")
    plt.show()
    Log.success()
