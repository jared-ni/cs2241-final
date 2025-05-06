import numpy as np
import mmh3

class CountMinSketch:
    def __init__(self, width: int, depth: int):
        self.width = width
        self.depth = depth
        self.count = np.zeros((depth, width), dtype=int)

    def update(self, idx: int, val: int):
        for i in range(self.depth):
            h = mmh3.hash(str(idx), i) % self.width
            self.count[i, h] += val

    def query(self, idx: int):
        estimates = []
        for i in range(self.depth):
            h = mmh3.hash(str(idx), i) % self.width
            estimates.append(self.count[i, h])
        return min(estimates)
