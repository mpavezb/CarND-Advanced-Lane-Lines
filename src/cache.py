import pickle
import os.path

from .logger import Log


class Cache:
    def __init__(self, fname):
        self.fname = "pickle/" + fname

    def exists(self):
        return os.path.isfile(self.fname)

    def save(self, data):
        Log.subsection("Saving data to pickle file: %s" % self.fname)
        pickle.dump(data, open(self.fname, "wb"))

    def load(self):
        Log.subsection("Loading data from pickle file: %s" % self.fname)
        data = pickle.load(open(self.fname, "rb"))
        return data
