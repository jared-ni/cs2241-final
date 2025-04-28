import numpy as np
from PIL import Image
from bitarray import bitarray
from helper import *
from bloom import BloomFilter
from options import *
from compressor_base import CompressorBase
from typing import Any, Callable


class Compressor(CompressorBase):
    def __init__(self,
                 image_options: ImageOptions,
                 quantization_options: QuantizationOptions | None,
                 segment_options: SegmentOptions | None,
                 linear_regression_options: LinearRegressionOptions | None,
                 delta_encoding_options: DeltaEncodingOptions | None,
                 bloom_filter_options: BloomFilterOptions | None,
    ):
        super().__init__()
        
        # Image settings
        image_options.validate()
        self.image_height = image_options.height
        self.image_width = image_options.width

        # Quantization settings
        if quantization_options:
            quantization_options.validate()
            self.quantization_bits = quantization_options.quantization_bits
        else:
            self.quantization_bits = 8

        # Segmenting settings
        if segment_options:
            segment_options.validate(self.image_height, self.image_width)
            self.segment_height = segment_options.height
            self.segment_width = segment_options.width
        else:
            self.segment_height = self.image_height
            self.segment_width = self.image_width

        # Linear regression settings
        if linear_regression_options:
            linear_regression_options.validate()
            self.do_linear_regression = True
        else:
            self.do_linear_regression = False
        
        # Delta encoding settings
        if delta_encoding_options:
            delta_encoding_options.validate()
            self.do_delta_encoding = True
            self.anchor_frequency = delta_encoding_options.anchor_frequency
            self.smear_bits = delta_encoding_options.smear_bits
        else:
            self.do_delta_encoding = False
        
        # Bloom filter settings
        if bloom_filter_options:
            bloom_filter_options.validate()
            self.do_bloom_filter = True
            ln_fpp = np.log(bloom_filter_options.fpp)
            ln2 = np.log(2)
            self.bloom_bits_per_element: float = -ln_fpp / ln2**2
            self.bloom_hash_count: int = round(-ln_fpp / ln2)
        else:
            self.do_bloom_filter = False

        # Set parameters derived from the ones above
        self.set_dependent_parameters()


    def compress_image(self, image_file: str, compressed_file: str, log_file: str | None = None):
        # Array of pixels making up the image
        arr = np.array(Image.open(image_file))
        assert len(arr.shape) == 3
        assert arr.shape[0] == self.image_height
        assert arr.shape[1] == self.image_width
        assert arr.shape[2] == 3
        assert (0 <= arr).all() and (arr <= 255).all()

        self.compressed_bitarray = bitarray()
        self.log_file = open(log_file, 'w') if log_file else None

        # We start by adding some metadata to the compressed file
        self.write(self.image_height, int2ba, 16, msg='Image height: ')
        self.write(self.image_width, int2ba, 16, msg='Image width: ')
        self.write(self.quantization_bits, int2ba, 4, msg='Quantization bits: ')
        self.write(self.segment_height, int2ba, 16, msg='Segment height: ')
        self.write(self.segment_width, int2ba, 16, msg='Segment width: ')
        self.write(int(self.do_linear_regression), int2ba, 1, msg='Linear regression? ')
        self.write(int(self.do_delta_encoding), int2ba, 1, msg='Delta encoding? ')
        self.write(int(self.do_bloom_filter), int2ba, 1, msg='Bloom filter? ')
        if self.do_delta_encoding:
            self.write(self.anchor_frequency, int2ba, 32, msg='Anchor frequency: ')
            self.write(self.smear_bits, int2ba, 4, msg='Smear bits: ')
        if self.do_bloom_filter:
            self.write(self.bloom_hash_count, int2ba, 8, msg='Bloom hash count: ')

        if self.do_delta_encoding:
            # Mask indicating locations of all the anchors
            anchor_mask = np.zeros(self.segment_size, dtype=bool)
            anchor_mask[::self.anchor_frequency] = 1
        
        # Quantize and segment the image
        arr = self.quantize(arr)
        segments = self.segment(arr)

        for segment_idx, segment in enumerate(segments):
            self.log_write(f'\n\n\nSEGMENT {segment_idx}\n')
            self.log_indent()

            red = segment[:, 0]
            green = segment[:, 1]
            blue = segment[:, 2]

            # Linear regression
            if self.do_linear_regression:
                # Regress green and blue channels on the red channel for each segment
                # Green and blue channels are then represented by their residuals
                green, green_beta0, green_beta1 = self.linear_regression(red, green)
                blue, blue_beta0, blue_beta1 = self.linear_regression(red, blue)
                # Add regression coefficients and intercepts to the compressed file
                self.write(green_beta0, float2ba, 32, True, msg='Green beta0: ')
                self.write(green_beta1, float2ba, 32, True, msg='Green beta1: ')
                self.write(blue_beta0, float2ba, 32, True, msg='Blue beta0: ')
                self.write(blue_beta1, float2ba, 32, True, msg='Blue beta1: ')
            
            for channel, channel_name in zip((red, green, blue), ('RED', 'GREEN', 'BLUE')):
                self.log_write(f'\n{channel_name} CHANNEL\n')
                self.log_indent()

                # Delta encoding
                if self.do_delta_encoding:
                    # Delta-encode and smear each channel
                    delta_arr = self.delta_encoding(channel)
                    delta_arr = self.smear(delta_arr)

                    # Add anchors to compressed file
                    anchors = delta_arr[anchor_mask]
                    self.write(anchors, intarr2ba, self.bits_per_anchor_val, self.is_signed_anchor,
                               msg='Anchors:\n')
                    self.log_write()

                    # Everything that is not an anchor is a difference ->
                    # handle these in next step
                    sparse_arr = delta_arr[~anchor_mask]

                else:  # No delta encoding -> continue to next step
                    sparse_arr = channel
                
                # Bloom filters
                if self.do_bloom_filter:
                    # Create Bloom filters
                    bloom_filters, mode = self.bloom_filter(sparse_arr)
                    # The mode from sparse_arr
                    self.write(mode, int2ba, self.bits_per_bloom_val, self.is_signed_bloom,
                               msg='Mode: ')
                    # The number of Bloom filters used
                    self.write(len(bloom_filters), int2ba, self.bits_per_bloom_val,
                               msg='Number of Bloom filters: ')
                    for num, bloom_filter in bloom_filters.items():
                        self.log_write('BLOOM FILTER')
                        self.log_indent()
                        # The value the Bloom filter represents
                        self.write(num, int2ba, self.bits_per_bloom_val, self.is_signed_bloom,
                                   msg='Bloom filter value: ')
                        # The number of bits in the Bloom filter
                        self.write(len(bloom_filter.bit_array), int2ba, 32,
                                   msg='Number of bits in Bloom filter: ')
                        # The Bloom filter
                        self.write(bloom_filter.bit_array,
                                   msg='Bloom filter:\n')
                        self.log_deindent()

                else:  # No Bloom filters -> directly write the sparse array
                    # Sparse array
                    self.write(sparse_arr, intarr2ba, self.bits_per_bloom_val, self.is_signed_bloom,
                               msg='Array:\n')
                
                self.log_deindent()
            
            self.log_deindent()

        with open(compressed_file, 'wb') as f:
            self.compressed_bitarray.tofile(f)
        
        if self.log_file:
            self.log_file.close()

    def write(self,
              val: Any,
              func: Callable | None = None,
              length: int | None = None,
              signed: bool = False,
              msg: Any = None):
        """
        Use `func` to convert `val` into a bitarray and write to the compressed file.
        If no `func` provided, then `val` must already be a bitarray.
        `val` (or if `val` is a list, each individual element of `val`) should be represented with `length` bits
        and be treated as either signed or unsigned depending on `signed`.
        If `msg` is provided and there is a log file, write `msg` followed by `val` to the log file.
        """
        self.compressed_bitarray += func(val, length, signed) if func else val
        if msg is not None:
            str_val = self.log_as_str(val)
            self.log_write(str(msg) + str_val)

    def quantize(self, arr: np.ndarray) -> np.ndarray:
        """
        Input:
        - arr: np.ndarray(height, width, channels)

        Output:
        - np.ndarray(height, width, channels),
          quantized to quantization_bits bits
        """
        return arr // self.quantization_factor

    def segment(self, arr: np.ndarray) -> list[np.ndarray]:
        """
        Input:
        - arr: np.ndarray(height, width, channels)
        - Assume image is 128 x 128, divisible by the segment size
        - Assume segment size: 

        Output:
        - list of np.ndarray(segment_size, channels)

        Each numpy array in the output is an image segment, flattened.
        Each segment represents a rectangle with dimensions segment_height by segment_width.
        """

        image_height, image_width, image_channels = arr.shape
        segments = []

        # iterate over image grid, one segment at a time
        for row_start in range(0, image_height, self.segment_height):
            for col_start in range(0, image_width, self.segment_width):
                # get end row and end col
                # if size doesn't match, image size not divisible by segment size
                row_end = min(row_start + self.segment_height, image_height)
                col_end = min(col_start + self.segment_width, image_width)
                
                segment_arr = arr[row_start:row_end, col_start:col_end, :]
                
                # make sure current segment is valid
                if segment_arr.size > 0:
                    flattened_segment_arr = segment_arr.reshape(-1, image_channels)
                    segments.append(flattened_segment_arr)
        
        return segments


    def linear_regression(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
        """
        Input:
        - x: np.ndarray(segment_size) -- predictor
        - y: np.ndarray(segment_size) -- objective

        Output:
        - res: np.ndarray(segment_size) -- residuals of y
        - beta0: float -- linear regression intercept
        - beta1: float -- linear regression coefficient
        """
        x = x.astype(float)
        y = y.astype(float)
        xbar = x.mean()
        ybar = y.mean()
        beta1: float = ((x-xbar)*(y-ybar)).sum() / ((x-xbar)**2).sum()
        beta0: float = ybar - beta1 * xbar
        yhat = beta0 + beta1 * x
        yhat = yhat.clip(0, self.quantization_max).astype(int)
        res = (y - yhat).astype(int)
        return res, beta0, beta1

    def delta_encoding(self, arr: np.ndarray) -> np.ndarray:
        """
        Input:
        - arr: np.ndarray(segment_size)

        Output:
        - np.ndarray(segment_size)
        """
        enc = np.ndarray(arr.shape, int)
        enc[1:] = np.diff(arr.astype(int))
        enc[::self.anchor_frequency] = arr[::self.anchor_frequency]
        return enc

    def smear(self, diff_arr: np.ndarray) -> np.ndarray:
        """
        Input:
        - arr: np.ndarray(segment_size)
        -- Anchors come at consistent interval: anchor_frequency
        -- At anchors, set carry to 0. 

        Output:
        - np.ndarray(segment_size)

        Happens ONLY AFTER DELTA-ENCODING. 

        """
        smeared_arr = np.zeros(self.segment_size, dtype=int)
        carry = 0

        for i in range(diff_arr.shape[0]):
            if i % self.anchor_frequency == 0:
                carry = 0
                
            current_diff = diff_arr[i]
            carried_diff = current_diff + carry

            clipped_diff = np.clip(carried_diff, 
                                -self.smear_max,
                                self.smear_max)
            smeared_arr[i] = clipped_diff

            carry = carried_diff - clipped_diff
        
        return smeared_arr

        # doing difference array increase the number of bloom filters,
        # thus we use smearing to reduce the range of neg/pos numbers
    

    def bloom_filter(self, sparse_arr: np.ndarray) -> tuple[dict[int, BloomFilter], int]:
        """
        Input:
        - arr: np.ndarray

        Output:
        - ???

        background removal: don't make a bloom filter for the background mode

        returns 
        """
        num_to_indices: dict[int, list[int]] = {}
            
        for i, num in enumerate(sparse_arr):
            num = int(num)
            if num not in num_to_indices:
                num_to_indices[num] = []
            num_to_indices[num].append(i)
        
        # now we have the indices, initialize bloom filters
        # skip the most popular num
        max_num_count = float('-inf')
        max_num = float('inf')
        for num in num_to_indices:
            if len(num_to_indices[num]) > max_num_count:
                max_num_count = len(num_to_indices[num])
                max_num = num
            
        bloom_filters: dict[int, BloomFilter] = {}
        for num in num_to_indices:
            if num == max_num:
                continue
            size = round(len(num_to_indices[num]) * self.bloom_bits_per_element)
            bloom_filters[num] = BloomFilter(size, self.bloom_hash_count)
            # add every index in num_to_indices[num] to the new Bloom filter
            for index in num_to_indices[num]:
                bloom_filters[num].add(index)           

        return bloom_filters, max_num

        # assume dictionary: 
        # {1: bloomFilter, 2: bloomFilter, 3: bloomFilter, 4: bloomFilter}


compressor = Compressor(
    image_options=ImageOptions(height=256, width=256),
    quantization_options=QuantizationOptions(quantization_bits=2),
    segment_options=SegmentOptions(height=64, width=64),
    linear_regression_options=LinearRegressionOptions(),
    delta_encoding_options=DeltaEncodingOptions(anchor_frequency=20, smear_bits=2),
    bloom_filter_options=BloomFilterOptions(fpp=0.0001)
)
compressor.compress_image('imagenet-sample-images-resized/n01443537_goldfish.JPEG', 'compressed')
