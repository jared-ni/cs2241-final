import numpy as np
from PIL import Image
from bitarray.util import int2ba
from helper import *
from bloom import BloomFilter
from options import *


class Compressor:
    def __init__(self,
                 quantization_bits,
                 segment_options: SegmentOptions | None,
                 linear_regression_options: LinearRegressionOptions | None,
                 delta_encoding_options: DeltaEncodingOptions | None,
                 bloom_filter_options: BloomFilterOptions | None,
    ):
        # Quantization settings
        self.quantization_bits = quantization_bits
        self.quantization_max = 2**quantization_bits - 1
        self.quantization_factor = 256 // 2**quantization_bits

        # Segmenting settings
        if segment_options:
            self.do_segment = True
            self.segment_height = segment_options.height
            self.segment_width = segment_options.width
            self.segment_size = self.segment_height * self.segment_width
        else:
            self.do_segment = False

        # Linear regression settings
        if linear_regression_options:
            self.do_linear_regression = True
        else:
            self.do_linear_regression = False
        
        # Delta encoding settings
        if delta_encoding_options:
            self.do_delta_encoding = True
            self.anchor_frequency = delta_encoding_options.anchor_frequency
            self.anchor_count = self.segment_size // self.anchor_frequency
            self.diff_count = self.segment_size - self.anchor_count
            self.smear_bits = delta_encoding_options.smear_bits
            self.smear_max = 2**(self.smear_bits-1)-1
        else:
            self.do_delta_encoding = False
        
        # Bloom filter settings
        if bloom_filter_options:
            self.do_bloom_filter = True
            ln_fpp = np.log(bloom_filter_options.fpp)
            ln2 = np.log(2)
            self.bloom_bits_per_element = -ln_fpp / ln2**2
            self.bloom_hash_count = round(-ln_fpp / ln2)
        else:
            self.do_bloom_filter = False


    def compress_image(self, image_file, compressed_file):
        # Array of pixels making up the image
        arr = np.array(Image.open(image_file))
        height, width, channels = arr.shape

        # `compressed` is the raw string of bits that will make up the compressed file
        # We start by adding some metadata
        compressed = (
            int2ba(height, 16) + int2ba(width, 16) + int2ba(self.quantization_bits, 8) +
            int2ba(int(self.do_segment), 1) + int2ba(int(self.do_linear_regression), 1) +
            int2ba(int(self.do_delta_encoding), 1) + int2ba(int(self.do_bloom_filter), 1)
        )
        if self.do_segment:
            compressed += int2ba(self.segment_height, 16) + int2ba(self.segment_width, 16)
        if self.do_delta_encoding:
            compressed += int2ba(self.anchor_frequency, 32) + int2ba(self.smear_bits, 8)
        if self.do_bloom_filter:
            compressed += int2ba(self.bloom_hash_count, 8)

        # anchor_bits_needed: Number of bits needed to represent each non-delta-encoded value in image
        # anchor_is_signed: Whether these values are signed or unsigned
        if self.do_linear_regression:
            # Linear regression produces residuals, which are signed and so require an extra bit
            anchor_bits_needed = self.quantization_bits + 1
            anchor_is_signed = True
        else:
            anchor_bits_needed = self.quantization_bits
            anchor_is_signed = False
        # Add this metadata
        compressed += int2ba(anchor_bits_needed, 7) + int2ba(int(anchor_is_signed), 1)

        # bloom_bits_needed: Number of bits needed to represent each value in the Bloom filter step
        # bloom_is_signed: Whether these values are signed or unsigned
        if self.do_delta_encoding:
            # If we use delta encoding, the Bloom filter step sees the smeared differences
            bloom_bits_needed = self.smear_bits
            bloom_is_signed = True
        else:
            # If we don't use delta encoding, every value is a non-delta-encoded value
            bloom_bits_needed = anchor_bits_needed
            bloom_is_signed = anchor_is_signed
        # Add this metadata
        compressed += int2ba(bloom_bits_needed, 7) + int2ba(int(bloom_is_signed), 1)
        
        # Quantize and segment the image
        arr = self.quantize(arr)
        segments = self.segment(arr) if self.do_segment else [arr]

        for segment in segments:
            red = segment[:, 0].flatten()
            green = segment[:, 1].flatten()
            blue = segment[:, 2].flatten()

            # Linear regression
            if self.do_linear_regression:
                # Regress green and blue channels on the red channel for each segment
                # Green and blue channels are then represented by their residuals
                green, green_beta0, green_beta1 = self.linear_regression(red, green)
                blue, blue_beta0, blue_beta1 = self.linear_regression(red, blue)
                # Add regression coefficients and intercepts to the compressed file
                compressed += (
                    float2ba(green_beta0) + float2ba(green_beta1) +
                    float2ba(blue_beta0) + float2ba(blue_beta1)
                )
            
            for channel in (red, green, blue):
                # Delta encoding
                if self.do_delta_encoding:
                    # Delta-encode and smear each channel
                    delta_arr = self.delta_encoding(channel)
                    delta_arr = self.smear(delta_arr)

                    # Add anchors to compressed file
                    anchors = delta_arr[::self.anchor_frequency]
                    compressed += intarr2ba(anchors, anchor_bits_needed, signed=anchor_is_signed)

                    # Everything that is not an anchor is a difference ->
                    # handle these in next step
                    diff_mask = np.ones_like(delta_arr)
                    diff_mask[::self.anchor_frequency] = 0
                    diffs = delta_arr[diff_mask]
                    sparse_arr = diffs

                else:  # No delta encoding -> continue to next step
                    sparse_arr = channel
                
                # Bloom filters
                if self.do_bloom_filter:
                    # Create Bloom filters
                    bloom_filters = self.bloom_filter(sparse_arr)
                    for num, bloom_filter in bloom_filters.items():
                        # The value the Bloom filter represents
                        compressed += int2ba(num, bloom_bits_needed, signed=bloom_is_signed)
                        # The length of the Bloom filter
                        compressed += int2ba(len(bloom_filter.bit_array), 32)
                        # The Bloom filter
                        compressed += bloom_filter.bit_array

                else:  # No Bloom filters -> directly write the sparse array
                    compressed += intarr2ba(sparse_arr, bloom_bits_needed, signed=bloom_is_signed)

        with open(compressed_file, 'wb') as f:
            compressed.tofile(f)


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
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        xbar = x.mean()
        ybar = y.mean()
        beta1: np.float64 = ((x-xbar)*(y-ybar)).sum() / ((x-xbar)**2).sum()
        beta0: np.float64 = ybar - beta1 * xbar
        yhat = beta0 + beta1 * x
        yhat = yhat.clip(0, self.quantization_max)
        res = (y - yhat).astype(int)
        return res, beta0, beta1

    def delta_encoding(self, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        smeared_arr = np.zeros_like(diff_arr)
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
    

    def bloom_filter(self, sparse_arr: np.ndarray):
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

        return bloom_filters

        # assume dictionary: 
        # {1: bloomFilter, 2: bloomFilter, 3: bloomFilter, 4: bloomFilter}


quantization_bits = 4
segment_options = SegmentOptions(height=64, width=64)
linear_regression_options = LinearRegressionOptions()
delta_encoding_options = DeltaEncodingOptions(anchor_frequency=100, smear_bits=4)
bloom_filter_options = BloomFilterOptions(fpp=0.01)
compressor = Compressor(quantization_bits,
                        segment_options,
                        linear_regression_options,
                        delta_encoding_options,
                        bloom_filter_options)
compressor.compress_image('imagenet-sample-images-resized/n01443537_goldfish.JPEG', 'compressed')