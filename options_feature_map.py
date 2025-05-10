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
        assert 1 <= self.quantization_bits and self.quantization_bits <= 16


@dataclass
class CountMinOptions:
    bloom_fpp: float | None
    cm_epsilon: float
    cm_delta: float

    def validate(self):
        assert isinstance(self.bloom_fpp, float) or self.bloom_fpp is None
        assert isinstance(self.cm_epsilon, float)
        assert isinstance(self.cm_delta, float)
        assert self.bloom_fpp is None or 0 < self.bloom_fpp and self.bloom_fpp < 1
        assert self.cm_epsilon > 0
        assert 0 < self.cm_delta and self.cm_delta < 1


@dataclass
class BloomierOptions:
    second_table: bool
    fpp: float
    slots_per_key: int | float
    hash_count: int

    def validate(self):
        assert isinstance(self.second_table, bool)
        assert isinstance(self.fpp, float)
        assert isinstance(self.slots_per_key, int) or isinstance(self.slots_per_key, float)
        assert isinstance(self.hash_count, int)
        assert 0 < self.fpp and self.fpp < 1
        assert self.slots_per_key > 0
        assert self.hash_count > 0


@dataclass
class HuffmanOptions:
    symbol_size: int
    
    def validate(self):
        assert isinstance(self.symbol_size, int)
        assert 1 <= self.symbol_size and self.symbol_size <= 255
