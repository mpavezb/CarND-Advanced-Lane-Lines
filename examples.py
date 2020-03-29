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
from src.lane_tracker import draw_overlay


def ex_read(fname):
    Log.subsection("Processing image: " + fname)
    image = cv2.imread(fname)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_image


def ex_undistort(image, camera):
    Log.info("Distortion correction ...")
    undistorted = camera.undistort(image)
    return undistorted


def ex_edges(image):
    Log.info("Edge Detection ...")
    edge_detector = EdgeDetector()
    binary = edge_detector.detect(image)
    return binary, edge_detector


def ex_warp(warper, binary):
    # Warp Image
    Log.info("Perspective Transform ...")
    warped = warper.warp(binary)
    return warped


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
        image = ex_read(fname)
        undistorted = ex_undistort(image, camera)

        # Concatenate
        Log.info("Drawing ...")
        vis = np.concatenate((image, undistorted), axis=1)

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
        image = ex_read(fname)
        undistorted = ex_undistort(image, camera)
        _, edge_detector = ex_edges(undistorted)

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
    for idx, fname in enumerate(sorted(images)):
        image = ex_read(fname)
        undistorted = ex_undistort(image, camera)
        warped = ex_warp(warper, undistorted)

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

    out_fname = os.path.join("output_images", "warp.png")
    Log.info("Saving figure to %s" % out_fname)
    plt.savefig(out_fname)

    Log.subsection("Display")
    plt.show()


def RunLaneFittingExample():
    Log.section("Lane Fitting Example")
    camera = GetCalibratedCamera()
    warper = WarpMachine()
    images = glob.glob("test_images/*.jpg")

    f, axs = plt.subplots(len(images), 1, figsize=(20, 50))
    f.tight_layout()
    for idx, fname in enumerate(sorted(images)):
        image = ex_read(fname)
        undistorted = ex_undistort(image, camera)
        edges, _ = ex_edges(undistorted)
        warped = ex_warp(warper, edges)

        Log.info("Lane Fitting ...")
        lane_fitting = LaneFit(image.shape[1], image.shape[0])
        vis_lanes = lane_fitting.fit_polynomial(warped)

        # Curvature
        left_cr, right_cr = lane_fitting.get_curvature()
        pos = lane_fitting.get_vehicle_position()

        # Drawing
        Log.info("Drawing ...")
        pos_str = "Left" if pos < 0 else "Right"
        crl_text = "Radius of curvature (left) = %d m" % left_cr
        crr_text = "Radius of curvature (right) = %d m" % right_cr
        cr_text = "Radius of curvature (avg) = %d m" % ((left_cr + right_cr) / 2)
        pos_text = "Vehicle is %.1f m %s from the lane center" % (np.abs(pos), pos_str)
        Log.info(crl_text)
        Log.info(crr_text)
        Log.info(cr_text)
        Log.info(pos_text)

        def put_text(image, text, color=(0, 255, 255), ypos=100):
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, text, (350, ypos), font, 1, color, 2, cv2.LINE_AA)

        put_text(vis_lanes, crl_text, ypos=50)
        put_text(vis_lanes, crr_text, ypos=100)
        put_text(vis_lanes, cr_text, ypos=150)
        put_text(vis_lanes, pos_text, ypos=200)

        # Concatenate
        vis_edges = np.dstack((edges, edges, edges))
        vis_edges = vis_edges * np.max(undistorted)
        vis = np.concatenate((undistorted, vis_edges, vis_lanes), axis=1)

        # Display Comparison
        axs[idx].imshow(vis)
        axs[idx].axis("off")

        # Save
        save_image(vis, fname, "lane_fitting_")

    Log.subsection("Display")
    plt.show()


def RunFullPipelineExample():
    Log.section("Full Pipeline Example")
    camera = GetCalibratedCamera()
    warper = WarpMachine()
    images = glob.glob("test_images/*.jpg")

    f, axs = plt.subplots(len(images), 1, figsize=(20, 50))
    f.tight_layout()
    for idx, fname in enumerate(sorted(images)):
        image = ex_read(fname)
        undistorted = ex_undistort(image, camera)
        edges, _ = ex_edges(undistorted)
        warped = ex_warp(warper, edges)

        Log.info("Lane Fitting ...")
        lane_fitting = LaneFit(image.shape[1], image.shape[0])
        vis_lanes = lane_fitting.fit_polynomial(warped)

        # Curvature
        left_cr, right_cr = lane_fitting.get_curvature()
        pos = lane_fitting.get_vehicle_position()

        # Overlay
        Log.info("Create Overlay ...")
        vis_overlay = draw_overlay(warper, lane_fitting, undistorted, warped)

        # Concatenate
        vis_edges = np.dstack((edges, edges, edges))
        vis_edges = vis_edges * np.max(undistorted)
        vis_a = np.concatenate((undistorted, vis_edges), axis=1)
        vis_b = np.concatenate((vis_lanes, vis_overlay), axis=1)
        vis = np.concatenate((vis_a, vis_b), axis=0)

        # Display Comparison
        axs[idx].imshow(vis_overlay)
        axs[idx].axis("off")

        # Save
        save_image(vis, fname, "overlay_", "_debug")
        save_image(vis_overlay, fname, "overlay_")
        # break

    Log.subsection("Display")
    plt.show()
