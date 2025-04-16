import math
import mmh3

class BloomFilter:
    def __init__(self, size, hash_count):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = [0] * size

    def add(self, element):
        for i in range(self.hash_count):
            index = mmh3.hash(element, i) % self.size
            self.bit_array[index] = 1

    def check(self, element):
        for i in range(self.hash_count):
            index = mmh3.hash(element, i) % self.size
            if self.bit_array[index] == 0:
                return False
        return True

    @classmethod
    def optimal_size(cls, n, p):
      """
      Computes the optimal size of the bit array (m)
      given the expected number of elements (n) and desired false positive probability (p).
      """
      m = - (n * math.log(p)) / (math.log(2) ** 2)
      return int(m)

    @classmethod
    def optimal_hash_count(cls, m, n):
      """
      Computes the optimal number of hash functions (k)
      given the size of the bit array (m) and the expected number of elements (n).
      """
      k = (m / n) * math.log(2)
      return int(k)