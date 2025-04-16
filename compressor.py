import numpy as np
from PIL import Image
from bitarray.util import int2ba
from helper import *


class Compressor:
    def __init__(self,
                 quantization_granularity,
                 anchor_frequency,
                 bloom_fpp,
                 segment_height,
                 segment_width,):
        self.channel_granularity = quantization_granularity
        self.quantization_factor = 256 // quantization_granularity

        self.anchor_frequency = anchor_frequency

        self.bloom_fpp = bloom_fpp

        self.segment_height = segment_height
        self.segment_width = segment_width
        self.segment_size = segment_height * segment_width

        self.anchor_count = self.segment_size // anchor_frequency
        self.diff_count = self.segment_size - self.anchor_count


    def compress_image(self, image_file, compressed_file):
        arr = np.array(Image.open(image_file))
        height, width, channels = arr.shape

        bits = (
            int2ba(self.quantization_factor, 8) +
            int2ba(height, 16) + int2ba(width, 16) +
            int2ba(self.segment_height, 16) + int2ba(self.segment_width, 16) +
            int2ba(self.anchor_frequency, 32)
        )
        
        arr = self.quantize(arr)
        segments = self.segment(arr)
        for segment in segments:
            red = segment[:, :, 0].flatten()
            green = segment[:, :, 1].flatten()
            blue = segment[:, :, 2].flatten()
            green_res, green_beta0, green_beta1 = self.linear_regression(red, green)
            blue_res, blue_beta0, blue_beta1 = self.linear_regression(red, blue)
            
            bits += (
                float2ba(green_beta0) + float2ba(green_beta1) +
                float2ba(blue_beta0) + float2ba(blue_beta1)
            )
            
            for channel in (red, green_res, blue_res):
                anchors, diffs = self.delta_encoding(channel)
                diffs = self.smear(diffs)
                bloom_filters = self.bloom_filter(diffs)

                # TODO: determine number of bits
                # TODO: store Bloom filters
                bits += intarr2ba(anchors, ..., True)

        with open(compressed_file, 'wb') as f:
            bits.tofile(f)


    def quantize(self, arr):
        """
        Input:
        - arr: np.ndarray(height, width, channels)

        Output:
        - np.ndarray(height, width, channels),
          quantized to have only channel_granularity unique values
        """
        return arr // self.quantization_factor

    def segment(self, arr):
        """
        Input:
        - arr: np.ndarray(height, width, channels)

        Output:
        - list of np.ndarray(segment_size, channels),
          except possibly smaller on the boundaries of the image

        Each numpy array in the output is an image segment, flattened.
        """
        # TODO
        pass

    def linear_regression(self, x, y):
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
        res = (y - yhat).astype(np.int8)
        return res, beta0, beta1

    def delta_encoding(self, arr):
        """
        Input:
        - arr: np.ndarray(segment_size)

        Output:
        - anchors: np.ndarray(anchor_count)
        - diffs: np.ndarray(diff_count)
        """
        # TODO -- modify to match specification
        enc = np.ndarray(arr.shape, np.int8)
        enc[1:] = np.diff(arr.astype(np.int8))
        enc[::self.anchor_frequency] = arr[::self.anchor_frequency]
        return enc

    def smear(self, arr):
        """
        Input:
        - arr: np.ndarray(diff_count)

        Output:
        - np.ndarray(diff_count)
        """
        # TODO
        pass

    def bloom_filter(self):
        """
        Input:
        - arr: np.ndarray(diff_count)

        Output:
        - ???
        """
        # TODO: see bloom.py; modify bloom.py
        pass


class Decompressor:
    def __init__():
        pass

    def decompress_image(self, compressed_file, output_file):
        pass
