import time
from datetime import timedelta

from src.logger import Log


class Profiler(object):
    def __init__(self, name):
        self.name = name
        self.elapsed = 0
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def update(self):
        self.elapsed += time.time() - self.start_time

    def fmt_timedelta(self, delta):
        return str(delta).split(".")[0]

    def fmt_seconds(self, seconds):
        return self.fmt_timedelta(timedelta(seconds=seconds))

    def get_elapsed(self):
        return self.elapsed

    def get_elapsed_str(self):
        return self.fmt_seconds(self.elapsed)

    def display_elapsed(self, total):
        percent = 100 * self.elapsed / total
        name = self.name.ljust(30)
        elapsed = self.get_elapsed_str()
        Log.info("%s:  %s (%5.1f%%)" % (name, elapsed, percent))

    def display_processing_factor(self, original_secs):
        factor = self.elapsed / original_secs
        Log.info("Processing Time Factor = x%.1f" % factor)
        Log.info("    Original Duration  = %.2f s" % original_secs)
        Log.info("    Elapsed Time       = %.2f s" % self.elapsed)
