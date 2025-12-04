# modules/smoothing.py
from collections import deque
import numpy as np

class TemporalSmoother:
    def __init__(self, window=5):
        self.history = deque(maxlen=window)

    def push(self, density):
        self.history.append(density)
        return np.mean(self.history)
