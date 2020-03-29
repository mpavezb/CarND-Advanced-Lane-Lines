import pickle
import os

from .logger import Log
from .save import chmod_rw_all


class Cache:
    def __init__(self, fname):
        self.fname = "pickle/" + fname

    def exists(self):
        return os.path.isfile(self.fname)

    def save(self, data):
        Log.subsection("Saving data to pickle file: %s" % self.fname)
        pickle.dump(data, open(self.fname, "w+b"))
        chmod_rw_all(self.fname)

    def load(self):
        Log.subsection("Loading data from pickle file: %s" % self.fname)
        data = pickle.load(open(self.fname, "r+b"))
        return data
