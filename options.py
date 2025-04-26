from dataclasses import dataclass

@dataclass
class SegmentOptions:
    height: int
    width: int

@dataclass
class LinearRegressionOptions:
    pass

@dataclass
class DeltaEncodingOptions:
    anchor_frequency: int
    smear_bits: int

@dataclass
class BloomFilterOptions:
    fpp: float
