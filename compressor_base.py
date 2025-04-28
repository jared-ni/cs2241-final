import numpy as np
from bitarray import bitarray
from typing import Any

class CompressorBase:
    def __init__(self):
        self.image_height: int
        self.image_width: int

        self.quantization_bits: int

        self.segment_height: int
        self.segment_width: int

        self.do_linear_regression: bool

        self.do_delta_encoding: bool
        self.anchor_frequency: int
        self.smear_bits: int

        self.do_bloom_filter: bool
        self.bloom_hash_count: float

        self.indent = 0
        np.set_printoptions(threshold=np.inf)


    def set_dependent_parameters(self):
        self.quantization_max: int = 2**self.quantization_bits - 1
        self.quantization_factor: int = 256 // 2**self.quantization_bits

        self.segment_size = self.segment_height * self.segment_width

        if self.do_delta_encoding:
            self.smear_max: int = 2**(self.smear_bits-1)-1

        # bits_per_anchor_val: Number of bits needed to represent each non-delta-encoded value in image
        # is_signed_anchor: Whether these values are signed or unsigned
        if self.do_linear_regression:
            # Linear regression produces residuals, which are signed and so require an extra bit
            self.bits_per_anchor_val = self.quantization_bits + 1
            self.is_signed_anchor = True
        else:
            self.bits_per_anchor_val = self.quantization_bits
            self.is_signed_anchor = False

        # bits_per_bloom_val: Number of bits needed to represent each value in the Bloom filter step
        # is_signed_bloom: Whether these values are signed or unsigned
        if self.do_delta_encoding:
            # If we use delta encoding, the Bloom filter step sees the smeared differences
            self.bits_per_bloom_val = self.smear_bits
            self.is_signed_bloom = True
        else:
            # If we don't use delta encoding, every value is a non-delta-encoded value
            self.bits_per_bloom_val = self.bits_per_anchor_val
            self.is_signed_bloom = self.is_signed_anchor

    def log_write(self, msg: Any = ''):
        """Write to the log file, if there is a log file."""
        if self.log_file:
            for line in str(msg).split('\n'):
                self.log_file.write(4*self.indent*' ' + line + '\n')

    def log_as_str(self, obj: Any) -> str:
        if isinstance(obj, bitarray):
            return str(np.array(list(obj)))
        if isinstance(obj, float):
            return str(np.float32(obj))
        return str(obj)

    def log_indent(self):
        """Increase the indent in the log file."""
        self.indent += 1

    def log_deindent(self):
        """Decrease the indent in the log file."""
        self.indent -= 1
