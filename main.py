import os
import glob
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
    Log.section("Distortion Correction")

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
        out_fname = os.path.join(
            "output_images", "undistort_" + os.path.basename(fname)
        )
        Log.info("Saving undistorted image: %s" % out_fname)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_fname, vis)

    Log.subsection("Display")
    plt.show()
    Log.success()


def RunPerspectiveTransformExample(camera):
    Log.section("Perspective Transform")

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
        out_fname = os.path.join("output_images", "warp_" + os.path.basename(fname))
        Log.info("Saving warped image: %s" % out_fname)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_fname, vis)

    Log.subsection("Display")
    plt.show()
    Log.success()


def imshow(axs, posx, posy, image, title, gray=True):
    cmap = None
    if gray:
        cmap = "gray"
    axs[posx, posy].imshow(image, cmap=cmap)
    axs[posx, posy].set_title(title, fontsize=15)
    axs[posx, posy].axis("off")


def pipeline(image, camera, fname, display_and_save=False):

    # Apply distortion correction
    undistorted = camera.undistort(image)

    # gray
    gray = cv2.cvtColor(undistorted, cv2.COLOR_RGB2GRAY)

    ###### HLS ######
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(undistorted, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]

    # tune using test1 (light), test2 (dark)
    # also, make sure to remove the capo with the threshold
    thresh = (100, 255)
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1

    ###### GRADIENT ######
    sobel_kernel = 13

    # sobel
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # magnitude
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # direction
    absx = np.absolute(sobel_x)
    absy = np.absolute(sobel_y)
    sobel_dir = np.arctan2(absy, absx)

    # threshold sobel x
    # scale to 8bit
    thresh = (50, 255)
    abs_sobel = np.absolute(sobel_x)
    scaled_sobel_x = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    sobel_x_binary = np.zeros_like(scaled_sobel_x)
    sobel_x_binary[(scaled_sobel_x >= thresh[0]) & (scaled_sobel_x <= thresh[1])] = 1

    # threshold sobel y
    thresh = (50, 255)
    abs_sobel = np.absolute(sobel_y)
    scaled_sobel_y = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    sobel_y_binary = np.zeros_like(scaled_sobel_y)
    sobel_y_binary[(scaled_sobel_y >= thresh[0]) & (scaled_sobel_y <= thresh[1])] = 1

    # threshold magnitude
    thresh = (50, 255)
    scale_factor = np.max(sobel_mag) / 255
    scaled_sobel_mag = np.uint8(sobel_mag / scale_factor)
    sobel_mag_binary = np.zeros_like(scaled_sobel_mag)
    sobel_mag_binary[
        (scaled_sobel_mag >= thresh[0]) & (scaled_sobel_mag <= thresh[1])
    ] = 1

    # threshold direction
    theta_deg = 60
    delta_deg = 20
    theta = theta_deg * np.pi / 180.0
    delta = delta_deg * np.pi / 180.0
    thresh = (theta - delta, theta + delta)
    # print(thresh)
    sobel_dir_binary = np.zeros_like(sobel_dir)
    sobel_dir_binary[(sobel_dir >= thresh[0]) & (sobel_dir <= thresh[1])] = 1

    # combined
    sobel_xy_binary = np.zeros_like(sobel_x)
    sobel_xy_binary[(sobel_x_binary == 1) & (sobel_y_binary == 1)] = 1
    sobel_md_binary = np.zeros_like(sobel_mag)
    sobel_md_binary[(sobel_mag_binary == 1) & (sobel_dir_binary == 1)] = 1
    sobel_all_binary = np.zeros_like(sobel_x)
    sobel_all_binary[(sobel_xy_binary == 1) | (sobel_md_binary == 1)] = 1
    combined_binary = np.zeros_like(sobel_x)
    combined_binary[(sobel_all_binary == 1) | (s_binary == 1)] = 1

    if not display_and_save:
        return combined_binary

    # Plot the result
    f, axs = plt.subplots(4, 4, figsize=(10, 10))
    f.tight_layout()

    placeholder = np.zeros_like(gray)
    imshow(axs, 0, 0, undistorted, "Undistorted", gray=False)
    imshow(axs, 0, 1, gray, "Gray Image")
    imshow(axs, 0, 2, s_channel, "HLS: S Channel")
    imshow(axs, 0, 3, s_binary, "HLS: S Channel Binary")
    imshow(axs, 1, 0, scaled_sobel_x, "Sobel X")
    imshow(axs, 1, 1, scaled_sobel_y, "Sobel Y")
    imshow(axs, 1, 2, scaled_sobel_mag, "Sobel Mag")
    imshow(axs, 1, 3, sobel_dir, "Sobel Dir")
    imshow(axs, 2, 0, sobel_x_binary, "Sobel X Thresh")
    imshow(axs, 2, 1, sobel_y_binary, "Sobel Y  Thresh")
    imshow(axs, 2, 2, sobel_mag_binary, "Sobel Mag  Thresh")
    imshow(axs, 2, 3, sobel_dir_binary, "Sobel Dir  Thresh")
    imshow(axs, 3, 0, sobel_xy_binary, "Sobel X-Y Thresh Comb")
    imshow(axs, 3, 1, sobel_md_binary, "Sobel Mag-Dir Thresh Comb")
    imshow(axs, 3, 2, sobel_all_binary, "Sobel Final")
    imshow(axs, 3, 3, combined_binary, "Combined All")

    plt.subplots_adjust(left=0.0, right=1, top=0.9, bottom=0.0)

    out_fname = os.path.join("output_images", os.path.basename(fname))
    out_fname = out_fname[:-4] + ".png"
    print("Saving File to %s" % out_fname)
    plt.savefig(out_fname)
    return combined_binary


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


def main():
    Log.debug_enabled = False
    camera = RunCalibration(display=False)
    # RunDistortionCorrectionExample(camera)
    RunPerspectiveTransformExample(camera)

    # images = glob.glob("test_images/*.jpg")
    # fname = "test_images/straight_lines1.jpg"
    # binary = pipeline(fname, camera)

    # for fname in images:
    #     image = cv2.imread(fname)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #     # binary = pipeline(image, camera, fname)

    #     undistorted = camera.undistort(image)
    #     warp(undistorted)

    # wait for user to finish program
    # plt.waitforbuttonpress()


if __name__ == "__main__":
    main()
