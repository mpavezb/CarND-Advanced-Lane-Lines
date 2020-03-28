from src.logger import Log
from examples import *


def main():
    Log.debug_enabled = False
    # RunCalibrationExample()
    # RunDistortionCorrectionExample()
    # RunEdgeDetectionExample()
    # RunPerspectiveTransformExample()
    # RunLaneFittingExample()
    RunFullPipelineExample()


if __name__ == "__main__":
    main()
