import numpy as np
from PIL import Image
from bitarray import bitarray
from helper import *
from bloom import BloomFilter
from compressor_base import CompressorBase
from typing import Any, Callable


class Decompressor(CompressorBase):
    def __init__(self, bias_toward_mode: bool):
        super().__init__()
        self.bias_toward_mode = bias_toward_mode
        self.pos = 0
        np.random.seed(0)

    def decompress_image(self, compressed_file: str, output_file: str, log_file: str | None = None):
        self.compressed = bitarray()
        with open(compressed_file, 'rb') as f:
            self.compressed.fromfile(f)

        self.log_file = open(log_file, 'w') if log_file else None
        
        # Read metadata at start of file
        self.image_height = self.read(16, ba2int, msg='Image height: ')
        self.image_width = self.read(16, ba2int, msg='Image width: ')
        self.quantization_bits = self.read(4, ba2int, msg='Quantization bits: ')
        self.segment_height = self.read(16, ba2int, msg='Segment height: ')
        self.segment_width = self.read(16, ba2int, msg='Segment width: ')
        self.do_linear_regression = bool(self.read(1, ba2int, msg='Linear regression? '))
        self.do_delta_encoding = bool(self.read(1, ba2int, msg='Delta encoding? '))
        self.do_bloom_filter = bool(self.read(1, ba2int, msg='Bloom filter? '))
        if self.do_delta_encoding:
            self.anchor_frequency = self.read(32, ba2int, msg='Anchor frequency: ')
            self.smear_bits = self.read(4, ba2int, msg='Smear bits: ')
        if self.do_bloom_filter:
            self.bloom_hash_count = self.read(8, ba2int, msg='Bloom hash count: ')

        # Set parameters derived from the ones above
        self.set_dependent_parameters()
        num_segments = (self.image_height // self.segment_height) * (self.image_width // self.segment_width)
        self.sparse_arr_len = self.segment_size
        if self.do_delta_encoding:
            num_anchors = 1 + (self.segment_size-1) // self.anchor_frequency
            anchors_bits = num_anchors * self.bits_per_anchor_val
            self.sparse_arr_len -= num_anchors
            # Mask indicating locations of all the anchors
            anchor_mask = np.zeros(self.segment_size, dtype=bool)
            anchor_mask[::self.anchor_frequency] = 1
        sparse_arr_bits = self.sparse_arr_len * self.bits_per_bloom_val

        segments = []
        for segment_idx in range(num_segments):
            self.log_write(f'\n\n\nSEGMENT {segment_idx}\n')
            self.log_indent()

            # Read coefficients and intercepts if linear regression used
            if self.do_linear_regression:
                green_beta0 = self.read(32, ba2float, signed=True, msg='Green beta0: ')
                green_beta1 = self.read(32, ba2float, signed=True, msg='Green beta1: ')
                blue_beta0 = self.read(32, ba2float, signed=True, msg='Blue beta0: ')
                blue_beta1 = self.read(32, ba2float, signed=True, msg='Blue beta1: ')

            channels = []
            for channel_name in ('RED', 'GREEN', 'BLUE'):
                self.log_write(f'\n{channel_name} CHANNEL\n')
                self.log_indent()

                # Read anchors if delta encoding is used
                if self.do_delta_encoding:
                    anchors = self.read(anchors_bits, ba2intarr, self.bits_per_anchor_val, self.is_signed_anchor,
                                        msg='Anchors:\n')
                    self.log_write()

                # Undo Bloom filter step
                if self.do_bloom_filter:
                    bloom_filters: dict[int, BloomFilter] = {}
                    mode = self.read(self.bits_per_bloom_val, ba2int, signed=self.is_signed_bloom,
                                     msg='Mode: ')
                    num_bloom_filters = self.read(self.bits_per_bloom_val, ba2int,
                                                  msg='Number of Bloom filters: ')
                    for _ in range(num_bloom_filters):
                        self.log_write('BLOOM FILTER')
                        self.log_indent()
                        num = self.read(self.bits_per_bloom_val, ba2int, signed=self.is_signed_bloom,
                                        msg='Bloom filter value: ')
                        size = self.read(32, ba2int, msg='Number of bits in Bloom filter: ')
                        bloom_filter_bits = self.read(size, msg='Bloom filter:\n')
                        bloom_filter = BloomFilter(bloom_filter_bits, self.bloom_hash_count)
                        bloom_filters[num] = bloom_filter
                        self.log_deindent()
                    sparse_arr = self.bloom_filter_undo(bloom_filters, mode)
                else:
                    sparse_arr = self.read(sparse_arr_bits, ba2intarr, self.bits_per_bloom_val, self.is_signed_bloom,
                                           msg='Array:\n')

                # Undo delta encoding step
                if self.do_delta_encoding:
                    # Array containing diffs interspersed with anchors at correct locations
                    delta_arr = np.ndarray(self.segment_size)
                    delta_arr[anchor_mask] = anchors
                    delta_arr[~anchor_mask] = sparse_arr
                    # Undo delta encoding
                    channel = self.delta_encoding_undo(delta_arr)
                else:
                    channel = sparse_arr

                channels.append(channel)
                
                self.log_deindent()

            red, green, blue = channels

            # Undo linear regression step
            if self.do_linear_regression:
                green = self.linear_regression_undo(red, green, green_beta0, green_beta1)
                blue = self.linear_regression_undo(red, blue, blue_beta0, blue_beta1)

            segment = np.ndarray((self.segment_size, 3))
            segment[:, 0] = red
            segment[:, 1] = green
            segment[:, 2] = blue
            segments.append(segment)

            self.log_deindent()

        # Undo segmenting and quantization steps
        arr = self.segment_undo(segments)
        arr = self.quantize_undo(arr)

        # Save image to file
        Image.fromarray(arr.astype(np.uint8)).save(output_file)

        if self.log_file:
            self.log_file.close()

    def read(self,
             num_bits: int,
             func: Callable | None = None,
             length: int | None = None,
             signed: bool = False,
             msg: Any = None):
        bits = self.read_bits(num_bits)
        val = func(bits, length, signed) if func else bits
        if msg is not None:
            str_val = self.log_as_str(val)
            self.log_write(str(msg) + str_val)
        return val

    def read_bits(self, length: int) -> bitarray:
        """Read the next `length` bits from `self.compressed`."""
        bits = self.compressed[self.pos : self.pos+length]
        self.pos += length
        return bits
    
    def quantize_undo(self, arr: np.ndarray) -> np.ndarray:
        return arr * self.quantization_factor
    
    def segment_undo(self, segments: list[np.ndarray]) -> np.ndarray:
        segments_per_row = self.image_height // self.segment_height
        segments_per_col = self.image_width // self.segment_width
        segments_grid = np.array(segments).reshape(segments_per_col, segments_per_row, self.segment_height, self.segment_width, 3)
        arr = segments_grid.swapaxes(1, 2).reshape(self.image_height, self.image_width, 3)
        return arr

    def linear_regression_undo(self, x: np.ndarray, res: np.ndarray, beta0: float, beta1: float) -> np.ndarray:
        x = x.astype(np.float32)
        yhat = beta0 + beta1 * x
        yhat = yhat.clip(0, self.quantization_max).astype(int)
        return (yhat + res).astype(int)

    def delta_encoding_undo(self, arr: np.ndarray) -> np.ndarray:
        n = (len(arr) + self.anchor_frequency - 1) // self.anchor_frequency  # ceiling division
        # Pad arr to the next multiple of self.anchor_frequency, to permit reshaping
        padded = np.pad(arr, (0, n * self.anchor_frequency - len(arr)), constant_values=0)
        reshaped = padded.reshape(-1, self.anchor_frequency)
        # Find cumulative sum to undo the delta encoding, then trim the extra padding
        decoded = reshaped.cumsum(axis=1).flatten()[:len(arr)]
        return decoded

    def bloom_filter_undo(self, bloom_filters: dict[int, BloomFilter], mode: int) -> np.ndarray:
        arr = np.ndarray(self.sparse_arr_len)
        for i in range(self.sparse_arr_len):
            # Identify all potential values according to Bloom filters
            positives = []
            for num, bloom_filter in bloom_filters.items():
                if bloom_filter.check(i):
                    positives.append(num)
            if positives:
                if self.bias_toward_mode:
                    positives = np.array(positives)
                    # Distance between the mode and each value in the Bloom filter positives
                    dist_from_mode = np.abs(positives-mode)
                    # Get the positives with minimum distance from mode
                    closest_positives = positives[dist_from_mode == dist_from_mode.min()]
                    # Randomly choose one to assign
                    arr[i] = np.random.choice(closest_positives)
                else:
                    # Choose uniformly at random from Bloom filter positives
                    arr[i] = np.random.choice(positives)
            else:
                # Index not present in any Bloom filter -> default to mode
                arr[i] = mode
                
        return arr


decompressor = Decompressor(bias_toward_mode=True)
decompressor.decompress_image('compressed', 'decompressed.png')
