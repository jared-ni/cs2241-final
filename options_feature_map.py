from dataclasses import dataclass


@dataclass
class FeatureMapOptions:
    shape: tuple[int]

    def validate(self):
        assert isinstance(self.shape, tuple)


@dataclass
class QuantizationOptions:
    quantization_bits: int

    def validate(self):
        assert isinstance(self.quantization_bits, int)
        assert 1 <= self.quantization_bits and self.quantization_bits <= 8


@dataclass
class CountMinOptions:
    bloom_fpp: float
    cm_epsilon: float
    cm_delta: float

    def validate(self):
        assert isinstance(self.bloom_fpp, float)
        assert isinstance(self.cm_epsilon, float)
        assert isinstance(self.cm_delta, float)
        assert 0 < self.bloom_fpp and self.bloom_fpp < 1
        assert self.cm_epsilon > 0
        assert 0 < self.cm_delta and self.cm_delta < 1


@dataclass
class BloomierOptions:
    fpp: float
    slots_per_key: int | float
    hash_count: int

    def validate(self):
        assert isinstance(self.fpp, float)
        assert isinstance(self.slots_per_key, int) or isinstance(self.slots_per_key, float)
        assert isinstance(self.hash_count, int)
        assert 0 < self.fpp and self.fpp < 1
        assert self.slots_per_key > 0
        assert self.hash_count > 0


@dataclass
class HuffmanOptions:
    group_size: int
    
    def validate(self):
        assert isinstance(self.group_size, int)
        assert 1 <= self.group_size and self.group_size <= 255
