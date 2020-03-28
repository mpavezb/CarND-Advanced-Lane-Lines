import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.logger import Log
from examples import *


def imshow(axs, posx, posy, image, title, gray=True):
    cmap = None
    if gray:
        cmap = "gray"
    axs[posx, posy].imshow(image, cmap=cmap)
    axs[posx, posy].set_title(title, fontsize=15)
    axs[posx, posy].axis("off")


def main():
    Log.debug_enabled = False
    # RunCalibrationExample()
    # RunDistortionCorrectionExample()
    # RunEdgeDetectionExample()
    # RunPerspectiveTransformExample()
    RunLaneFittingExample()


if __name__ == "__main__":
    main()
