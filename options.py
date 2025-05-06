from dataclasses import dataclass


@dataclass
class ImageOptions:
    height: int
    width: int

    def validate(self):
        assert isinstance(self.height, int) and isinstance(self.width, int)
        assert self.height > 0 and self.width > 0


@dataclass
class QuantizationOptions:
    quantization_bits: int

    def validate(self):
        assert isinstance(self.quantization_bits, int)
        assert 1 <= self.quantization_bits and self.quantization_bits <= 8


@dataclass
class SegmentOptions:
    height: int
    width: int

    def validate(self, image_height: int,  image_width: int):
        assert isinstance(self.height, int) and isinstance(self.width, int)
        assert self.height > 0 and self.width > 0
        assert image_height % self.height == 0
        assert image_width % self.width == 0


@dataclass
class LinearRegressionOptions:
    def validate(self):
        pass


@dataclass
class DeltaEncodingOptions:
    anchor_frequency: int
    smear_bits: int

    def validate(self):
        assert isinstance(self.anchor_frequency, int)
        assert isinstance(self.smear_bits, int)
        assert self.anchor_frequency >= 1
        assert 1 <= self.smear_bits and self.smear_bits <= 8


@dataclass
class BloomFilterOptions:
    fpp: float

    def validate(self):
        assert isinstance(self.fpp, float)
        assert 0 < self.fpp and self.fpp < 1

@dataclass
class LZWOptions:
    bits_per_val: int

    def validate(self):
        assert isinstance(self.bits_per_val, int)
        assert 1 <= self.bits_per_val and self.bits_per_val <= 255

@dataclass
class HuffmanOptions:
    group_size: int
    
    def validate(self):
        assert isinstance(self.group_size, int)
        assert 1 <= self.group_size and self.group_size <= 255
