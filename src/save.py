import os

import cv2
from .logger import Log


def save_image(image, fname, prefix="", suffix=""):
    basename = os.path.basename(fname)
    name, ext = os.path.splitext(basename)

    out_name = os.path.join("output_images", prefix + name + suffix + ext)
    Log.info("Saving image: %s" % out_name)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_name, image)
