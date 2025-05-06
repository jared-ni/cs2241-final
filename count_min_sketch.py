import numpy as np
import mmh3

class CountMinSketch:
    def __init__(self, width: int, depth: int, table: np.ndarray | None = None):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=int) if table is None else table

    def update(self, idx: int, val: int):
        for i in range(self.depth):
            h = mmh3.hash(str(idx), i) % self.width
            self.table[i, h] += val

    def query(self, idx: int):
        estimates = []
        for i in range(self.depth):
            h = mmh3.hash(str(idx), i) % self.width
            estimates.append(self.table[i, h])
        return min(estimates)
