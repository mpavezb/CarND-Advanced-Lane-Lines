from src.logger import Log
from src.recorder import Recorder
from src.calibration import CalibrationParameters, CameraModel


def RunCalibration(display=True):
    Log.section("Camera Calibration")

    params = CalibrationParameters()
    params.display = display

    camera = CameraModel(params)
    camera.calibrate()
    objpoints, imgpoints = camera.get_calibration()

    Recorder.save_calibration(objpoints, imgpoints)
    return objpoints, imgpoints


def main():
    objpoints, imgpoints = RunCalibration(display=False)

    # wait for user to finish program
    # plt.waitforbuttonpress()


if __name__ == "__main__":
    main()
