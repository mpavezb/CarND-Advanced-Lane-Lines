import pickle
import os
import stat

from .logger import Log


class Cache:
    def __init__(self, fname):
        self.fname = "pickle/" + fname

    def exists(self):
        return os.path.isfile(self.fname)

    def chmod_all(self):
        os.chmod(
            self.fname,
            stat.S_IRUSR
            | stat.S_IWUSR
            | stat.S_IRGRP
            | stat.S_IWGRP
            | stat.S_IROTH
            | stat.S_IWOTH,
        )

    def save(self, data):
        Log.subsection("Saving data to pickle file: %s" % self.fname)
        pickle.dump(data, open(self.fname, "w+b"))
        self.chmod_all()

    def load(self):
        Log.subsection("Loading data from pickle file: %s" % self.fname)
        data = pickle.load(open(self.fname, "r+b"))
        return data
