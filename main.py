from src.logger import Log
from src.calibration import CalibrationParameters, CameraModel


def RunCalibration(display=True):
    Log.section("Camera Calibration")

    params = CalibrationParameters()
    params.display = display

    camera = CameraModel(params)
    camera.calibrate()

    # getting the points
    objpoints, imgpoints = camera.get_3d_to_2d_points()

    return camera


def main():
    Log.debug_enabled = False

    camera = RunCalibration(display=False)

    h = 100
    w = 200
    camera.get_calibration(w, h)

    # wait for user to finish program
    # plt.waitforbuttonpress()


if __name__ == "__main__":
    main()
