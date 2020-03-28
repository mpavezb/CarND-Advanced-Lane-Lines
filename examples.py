import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.logger import Log
from src.calibration import GetCalibratedCamera, WarpMachine
from src.filtering import EdgeDetector
from src.lane_fitting import LaneFit
from src.save import save_image


def RunCalibrationExample():
    Log.section("Camera Calibration Example")
    camera = GetCalibratedCamera()
    camera.display_calibration()


def RunDistortionCorrectionExample():
    Log.section("Distortion Correction Example")
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
    for idx, fname in enumerate(sorted(images)):
        # Read
        Log.subsection("Processing image: " + fname)
        img_ = cv2.imread(fname)
        img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)

        # Apply distortion correction
        Log.info("Distortion correction ...")
        undistorted = camera.undistort(img)

        # Concatenate
        Log.info("Drawing ...")
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


def RunEdgeDetectionExample():
    Log.section("EdgeDetection")
    camera = GetCalibratedCamera()
    images = glob.glob("test_images/*.jpg")

    f, axs = plt.subplots(len(images), 1, figsize=(20, 70))
    f.tight_layout()
    for idx, fname in enumerate(sorted(images)):
        # Read
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
        Log.info("Drawing ...")
        edge_detector.display_pipeline(fname, axs[idx])

    Log.subsection("Display")
    plt.show()


def RunPerspectiveTransformExample():
    Log.section("Perspective Transform Example")
    camera = GetCalibratedCamera()
    warper = WarpMachine()
    images = glob.glob("test_images/*.jpg")

    f, axs = plt.subplots(len(images), 1, figsize=(20, 50))
    f.tight_layout()
    axs[0].set_title("Undistorted and Warped Images", fontsize=15)
    for idx, fname in enumerate(sorted(images)):
        # Read
        Log.subsection("Processing image: " + fname)
        image = cv2.imread(fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply distortion correction
        Log.info("Distortion correction ...")
        undistorted = camera.undistort(image)

        # Warp Image
        Log.info("Perspective Transform ...")
        warped = warper.warp(undistorted)

        # Draw
        Log.info("Drawing ...")
        warper.draw_src(undistorted)
        warper.draw_dst(warped)

        # Concatenate
        vis = np.concatenate((undistorted, warped), axis=1)

        # Display Comparison
        axs[idx].imshow(vis)
        axs[idx].axis("off")

        # Save
        save_image(vis, fname, "warp_")

    Log.subsection("Display")
    plt.show()


def RunLaneFittingExample():
    Log.section("Lane Fitting Example")
    camera = GetCalibratedCamera()
    warper = WarpMachine()
    images = glob.glob("test_images/*.jpg")

    f, axs = plt.subplots(len(images), 1, figsize=(20, 50))
    f.tight_layout()
    axs[0].set_title("Undistorted Image and Lane Fitting", fontsize=15)
    for idx, fname in enumerate(sorted(images)):
        # Read
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

        # Warp Image
        Log.info("Perspective Transform ...")
        warped = warper.warp(binary)

        # Lane Fitting
        Log.info("Lane Fitting ...")
        lane_fitting = LaneFit(image.shape[1], image.shape[0])
        _, _, out_img = lane_fitting.fit_polynomial(warped)

        # Lane Curvature
        left_cr_px, right_cr_px = lane_fitting.get_curvature_px()
        left_cr, right_cr = lane_fitting.get_curvature()

        pos = lane_fitting.get_vehicle_position()
        pos_str = "Left" if pos < 0 else "Right"
        crl_text = "Radius of curvature (left) = %d m" % left_cr
        crr_text = "Radius of curvature (right) = %d m" % right_cr
        cr_text = "Radius of curvature (avg) = %d m" % ((left_cr + right_cr) / 2)
        pos_text = "Vehicle is %.1f m %s from the lane center" % (np.abs(pos), pos_str)
        Log.info(crl_text)
        Log.info(crr_text)
        Log.info(cr_text)
        Log.info(pos_text)

        # Drawing
        Log.info("Drawing ...")

        def put_text(image, text, color=(0, 255, 255), ypos=100):
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, text, (350, ypos), font, 1, color, 2, cv2.LINE_AA)

        put_text(out_img, crl_text, ypos=50)
        put_text(out_img, crr_text, ypos=100)
        put_text(out_img, cr_text, ypos=150)
        put_text(out_img, pos_text, ypos=200)

        # Concatenate
        vis = np.concatenate((undistorted, out_img), axis=1)

        # Display Comparison
        axs[idx].imshow(vis)
        axs[idx].axis("off")

        # Save
        save_image(vis, fname, "lane_fitting_")

    Log.subsection("Display")
    plt.show()
