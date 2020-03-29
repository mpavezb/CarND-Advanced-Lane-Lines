from src.logger import Log
from src.lane_tracker import LaneLinesTracker
from examples import *


def ProcessProjectVideo(subclip_seconds=None):
    Log.section("Project Video")

    input_file = "project_video.mp4"
    output_file = "output_videos/project_video.mp4"

    tracker = LaneLinesTracker()
    clip = tracker.process_video(input_file, output_file, subclip_seconds)
    return output_file


def main():
    Log.debug_enabled = False
    # RunCalibrationExample()
    # RunDistortionCorrectionExample()
    # RunEdgeDetectionExample()
    # RunPerspectiveTransformExample()
    # RunLaneFittingExample()
    RunFullPipelineExample()

    # ProcessProjectVideo(subclip_seconds=0.5)


if __name__ == "__main__":
    main()
