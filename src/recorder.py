import pickle

from .logger import Log


class Recorder:
    def save_calibration(objpoints, imgpoints):
        fname = "pickle/calibration.p"
        Log.subsection("Saving calibration data to pickle file: %s" % fname)
        data = dict()
        data["objpoints"] = objpoints
        data["imgpoints"] = imgpoints
        pickle.dump(data, open(fname, "wb"))
        Log.success()
