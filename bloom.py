import mmh3
from bitarray import bitarray

class BloomFilter:
    def __init__(self, size, hash_count, bits: bitarray | None = None):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bits if bits else bitarray([0] * size)

    def add(self, element):
        for i in range(self.hash_count):
            index = mmh3.hash(str(element), i) % self.size
            self.bit_array[index] = 1

    def check(self, element):
        for i in range(self.hash_count):
            index = mmh3.hash(str(element), i) % self.size
            if self.bit_array[index] == 0:
                return False
        return True
