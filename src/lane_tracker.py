import cv2
import numpy as np
from moviepy.editor import VideoFileClip

from .logger import Log
from .calibration import GetCalibratedCamera, WarpMachine
from .filtering import EdgeDetector
from .lane_fitting import LaneFit
from .save import chmod_rw_all, delete_file
from .profiler import Profiler


def draw_overlay(warper, lane_fitting, undistorted, warped):
    # get curvature and vehicle position
    left_cr, right_cr = lane_fitting.get_curvature()
    pos = lane_fitting.get_vehicle_position()

    # get fitpoints
    pts_y, left_fitx, right_fitx = lane_fitting.get_fitpoints()

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, pts_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, pts_y])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    overlay = warper.unwarp(color_warp)

    # Combine the result with the original image
    vis_overlay = cv2.addWeighted(undistorted, 1, overlay, 0.3, 0)

    pos_str = "Left" if pos < 0 else "Right"
    crl_text = "Radius of curvature (left) = %.1f km" % (left_cr / 1000)
    crr_text = "Radius of curvature (right) = %.1f km" % (right_cr / 1000)
    cr_text = "Radius of curvature (avg) = %.1f km" % ((left_cr + right_cr) / 2000)
    pos_text = "Vehicle is %.1f m %s from the lane center" % (np.abs(pos), pos_str)

    def put_text(image, text, color=(255, 255, 255), ypos=100):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, text, (350, ypos), font, 1, color, 2, cv2.LINE_AA)

    put_text(vis_overlay, crl_text, ypos=50)
    put_text(vis_overlay, crr_text, ypos=100)
    put_text(vis_overlay, cr_text, ypos=150)
    put_text(vis_overlay, pos_text, ypos=200)

    return vis_overlay


class LaneLinesTracker(object):
    def __init__(self):
        self.camera = GetCalibratedCamera()
        self.warper = WarpMachine()

        # profiling
        self.p_video = Profiler("Total Time")
        self.p_undistort = Profiler("Distortion  Correction")
        self.p_edges = Profiler("Edge Detection")
        self.p_warp = Profiler("Perspective Transform")
        self.p_fitting = Profiler("Lane Fitting")
        self.p_overlay = Profiler("Overlay Drawing")

    def process_video(self, input_file, output_file, subclip_seconds=None):
        # delete output file to avoid permission problems between docker/user on write
        delete_file(output_file)

        self.p_video.start()

        # read
        Log.subsection("Reading video file: %s" % input_file)
        clip = VideoFileClip(input_file)

        # subclip
        if subclip_seconds:
            Log.info("Clipping video to: %.1f s" % subclip_seconds)
            clip = clip.subclip(0, subclip_seconds)

        # set image handler
        Log.info("Setting Image Handler ...")
        clip = clip.fl_image(self.process_image)

        # process / save
        Log.subsection("Processing Video ...")
        clip.write_videofile(output_file, audio=False, verbose=False)
        chmod_rw_all(output_file)
        self.p_video.update()

        # display profiling results
        Log.subsection("Profiling Results ...")
        total_secs = self.p_video.get_elapsed()
        self.p_video.display_elapsed(total_secs)
        self.p_undistort.display_elapsed(total_secs)
        self.p_edges.display_elapsed(total_secs)
        self.p_warp.display_elapsed(total_secs)
        self.p_fitting.display_elapsed(total_secs)
        self.p_overlay.display_elapsed(total_secs)
        self.p_video.display_processing_factor(clip.duration)

    def process_image(self, image):
        # Distortion correction
        self.p_undistort.start()
        undistorted = self.camera.undistort(image)
        self.p_undistort.update()

        # Edge Detection
        self.p_edges.start()
        edge_detector = EdgeDetector()
        edges = edge_detector.detect(undistorted)
        self.p_edges.update()

        # Perspective Transform
        self.p_warp.start()
        warped = self.warper.warp(edges)
        self.p_warp.update()

        # Lane Fitting
        self.p_fitting.start()
        lane_fitting = LaneFit(image.shape[1], image.shape[0])
        vis_lanes = lane_fitting.fit_polynomial(warped)
        self.p_fitting.update()

        # Draw Overlay
        self.p_overlay.start()
        vis_overlay = draw_overlay(self.warper, lane_fitting, undistorted, warped)
        self.p_overlay.update()

        return vis_overlay
