import mmh3
from bitarray import bitarray

class BloomFilter:
    def __init__(self, bits: int | bitarray, hash_count: int):
        self.hash_count = hash_count
        if isinstance(bits, int):
            self.size = bits
            self.bit_array = bitarray([0] * bits)
        elif isinstance(bits, bitarray):
            self.size = len(bits)
            self.bit_array = bits
        else:
            raise Exception

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
