# compressor_jpeg.py

import numpy as np
from PIL import Image
import cv2
import scipy.fftpack
from math import ceil
import pickle
from typing import Any, Dict, Tuple, List, Optional, Union # <-- Make sure Optional is here
import collections
import heapq
from bitarray import bitarray
from bitarray.util import ba2int, int2ba

class ImageOptions:
    def __init__(self, height: int, width: int): self.height = height; self.width = width
    def validate(self): pass
class JpegOptions:
    def __init__(self, quality: int = 75, subsampling_mode: str = '4:2:0'): self.quality = quality; self.subsampling_mode = subsampling_mode
    def validate(self): pass

class CompressorBase:
    def __init__(self): self._log_indent_level = 0; self.log_file = None
    def log_write(self, msg: str = ''):
        if self.log_file: print(f"{'  '*self._log_indent_level}{msg}", file=self.log_file); self.log_file.flush()
    def log_indent(self): self._log_indent_level += 1
    def log_deindent(self): self._log_indent_level = max(0, self._log_indent_level - 1)
    def set_dependent_parameters(self): pass

Q_LUM_ST = np.array([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],[14,13,16,24,40,57,69,56],[14,17,22,29,51,87,80,62],[18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],[49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]], dtype=np.int32)
Q_CHROM_ST = np.array([[17,18,24,47,99,99,99,99],[18,21,26,66,99,99,99,99],[24,26,56,99,99,99,99,99],[47,66,99,99,99,99,99,99],[99,99,99,99,99,99,99,99],[99,99,99,99,99,99,99,99],[99,99,99,99,99,99,99,99],[99,99,99,99,99,99,99,99]], dtype=np.int32)
zigzag_indices = np.array([[ 0, 1, 5, 6,14,15,27,28],[ 2, 4, 7,13,16,26,29,42],[ 3, 8,12,17,25,30,41,43],[ 9,11,18,24,31,40,44,53],[10,19,23,32,39,45,52,54],[20,22,33,38,46,51,55,60],[21,34,37,47,50,56,59,61],[35,36,48,49,57,58,62,63]])
zigzag_flat_indices = zigzag_indices.flatten(); zigzag_sort_indices = np.argsort(zigzag_flat_indices)


HuffmanNode = collections.namedtuple('HuffmanNode', ['symbol', 'freq', 'left', 'right'])
HuffmanNode.__lt__ = lambda x, y: x.freq < y.freq
def build_huffman_tree(frequencies: collections.Counter): # ... definition ...
    if not frequencies: return None
    heap = [HuffmanNode(symbol=s, freq=f, left=None, right=None) for s, f in frequencies.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        node1 = heapq.heappop(heap); node2 = heapq.heappop(heap)
        merged_freq = node1.freq + node2.freq
        left, right = (node1, node2) if node1.freq <= node2.freq else (node2, node1)
        merged_node = HuffmanNode(symbol=None, freq=merged_freq, left=left, right=right)
        heapq.heappush(heap, merged_node)
    return heap[0] if heap else None
def generate_huffman_codes(tree):
    codes = {};
    if tree is None: return codes
    def traverse(node, current_code):
        if node.symbol is not None: codes[node.symbol] = current_code if current_code else bitarray('0'); return
        if node.left: traverse(node.left, current_code + bitarray('0'))
        if node.right: traverse(node.right, current_code + bitarray('1'))
    traverse(tree, bitarray()); return codes
EOB = ('EOB',)


class JpegStyleCompressor(CompressorBase):
     def __init__(self, image_options: ImageOptions, jpeg_options: JpegOptions):
         super().__init__()
         image_options.validate(); jpeg_options.validate()
         self.image_height = image_options.height; self.image_width = image_options.width
         self.quality = jpeg_options.quality; self.subsampling_mode = jpeg_options.subsampling_mode
         self.set_dependent_parameters()

     def set_dependent_parameters(self):
         self.q_lum_scaled = self._scale_quantization_table(Q_LUM_ST, self.quality)
         self.q_chrom_scaled = self._scale_quantization_table(Q_CHROM_ST, self.quality)

     def _scale_quantization_table(self, q_table: np.ndarray, quality: int) -> np.ndarray:
         if quality < 1: quality = 1
         if quality > 100: quality = 100
         scale = 5000 / quality if quality < 50 else 200 - 2 * quality
         scaled_table = np.floor((q_table * scale + 50) / 100)
         return np.clip(scaled_table, 1, 255).astype(np.int32)

     def _pad_image(self, channel: np.ndarray, block_size: int = 8):
         h, w = channel.shape
         pad_h = (block_size - h % block_size) % block_size
         pad_w = (block_size - w % block_size) % block_size
         padded_channel = np.pad(channel, ((0, pad_h), (0, pad_w)), mode='edge')
         return padded_channel, (h, w)

     def _split_into_blocks(self, channel: np.ndarray, block_size: int = 8):
         h, w = channel.shape; blocks = []
         grid_h, grid_w = h // block_size, w // block_size
         for i in range(grid_h):
             for j in range(grid_w):
                 block = channel[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                 blocks.append(block)
         return blocks, (grid_h, grid_w)

     def _rgb_to_ycrcb(self, image_rgb: np.ndarray):
         if image_rgb.shape[:2] != (self.image_height, self.image_width):
             raise ValueError(f"Input image dimensions ({image_rgb.shape[0]}x{image_rgb.shape[1]}) != options ({self.image_height}x{self.image_width})")
         if image_rgb.ndim != 3 or image_rgb.shape[2] != 3: raise ValueError("Input must be HxWx3 RGB.")
         if image_rgb.dtype != np.uint8: image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
         image_ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
         y, cr, cb = cv2.split(image_ycrcb)
         return y, cb, cr

     def _chroma_subsample(self, cb: np.ndarray, cr: np.ndarray):
         h, w = cb.shape
         if self.subsampling_mode == '4:4:4': cb_sub, cr_sub = cb, cr
         elif self.subsampling_mode == '4:2:2':
             cb_sub = cv2.resize(cb, (w // 2, h), interpolation=cv2.INTER_LINEAR)
             cr_sub = cv2.resize(cr, (w // 2, h), interpolation=cv2.INTER_LINEAR)
         elif self.subsampling_mode == '4:2:0':
             cb_sub = cv2.resize(cb, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
             cr_sub = cv2.resize(cr, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
         else: raise ValueError(f"Unsupported subsampling: {self.subsampling_mode}")
         return cb_sub, cr_sub

     def _apply_dct_and_quantize_block(self, block: np.ndarray, q_table: np.ndarray):
         block_float = block.astype(np.float32) - 128.0
         dct_block = scipy.fftpack.dct(scipy.fftpack.dct(block_float.T, norm='ortho').T, norm='ortho')
         quantized_block = np.round(dct_block / q_table)
         return quantized_block.astype(np.int32)

     def _zigzag_scan(self, block_8x8: np.ndarray):
         if block_8x8.shape != (8, 8): raise ValueError("Input must be 8x8.")
         return block_8x8.flatten()[zigzag_sort_indices]

     def compress_image_raw(self, image_file: str, compressed_file: str, log_file: Optional[str] = None):
         pass


class JpegCompressorWithEntropy(JpegStyleCompressor):
    def __init__(self,
                 image_options: ImageOptions,
                 jpeg_options: JpegOptions
                 ):
        super().__init__(image_options, jpeg_options)

    def compress_image(self, image_file: str, compressed_file: str, log_file: Optional[str] = None):
        try:
            img = Image.open(image_file)
            if img.mode != 'RGB': img = img.convert('RGB')
            arr_rgb = np.array(img)
        except Exception as e: print(f"Error loading image: {e}"); return

        if log_file:
            try: self.log_file = open(log_file, 'w')
            except IOError as e: print(f"Warning: Could not open log file {log_file}: {e}"); self.log_file = None
        else: self.log_file = None

        self.log_write(f"--- Starting JPEG Compression (with RLE/Huffman) ---")
        self.log_indent()
        y, cb, cr = self._rgb_to_ycrcb(arr_rgb)
        cb_proc, cr_proc = self._chroma_subsample(cb, cr)
        y_proc = y
        self.log_deindent()

        all_symbols = {'y': [], 'cb': [], 'cr': []}
        encoded_data_bits = {'y': bitarray(), 'cb': bitarray(), 'cr': bitarray()}
        huffman_codes = {}
        grid_dims = {}
        original_shapes = {'y': y_proc.shape, 'cb': cb_proc.shape, 'cr': cr_proc.shape}

        self.log_write("\n--- Processing Channels (Pad, Block, DCT, Quantize) ---")
        for channel_data, q_table, name in [(y_proc, self.q_lum_scaled, 'y'),
                                             (cb_proc, self.q_chrom_scaled, 'cb'),
                                             (cr_proc, self.q_chrom_scaled, 'cr')]:
            self.log_write(f"\nProcessing {name.upper()} Channel...")
            self.log_indent()
            padded_channel, _ = self._pad_image(channel_data)
            blocks, grid_shape = self._split_into_blocks(padded_channel)
            grid_dims[name] = grid_shape

            channel_symbols = []
            previous_dc = 0

            self.log_write(f"Generating Symbols (DC diff + RLE AC)...")
            num_blocks = len(blocks)
            for i, block in enumerate(blocks):
                quantized_block = self._apply_dct_and_quantize_block(block, q_table)
                zigzag_coeffs = self._zigzag_scan(quantized_block).astype(np.int32)

                dc_coeff = zigzag_coeffs[0]; dc_diff = dc_coeff - previous_dc
                channel_symbols.append(dc_diff); previous_dc = dc_coeff

                ac_coeffs = zigzag_coeffs[1:]; run_length = 0
                for coeff in ac_coeffs:
                    if coeff == 0: run_length += 1
                    else: channel_symbols.append((run_length, coeff)); run_length = 0
                if run_length > 0 or np.all(ac_coeffs == 0): channel_symbols.append(EOB)

            all_symbols[name] = channel_symbols
            self.log_write(f"Generated {len(channel_symbols)} symbols.")
            self.log_deindent()

        self.log_write("\n--- Building Huffman Codes & Encoding Data ---")
        for name in ['y', 'cb', 'cr']:
            self.log_write(f"Encoding {name.upper()} Channel...")
            self.log_indent()
            channel_symbols = all_symbols[name]
            if not channel_symbols:
                self.log_write("No symbols.")
                huffman_codes[name] = {}
                encoded_data_bits[name] = bitarray()
                self.log_deindent()
                continue

            symbol_freqs = collections.Counter(channel_symbols)
            huff_tree = build_huffman_tree(symbol_freqs)
            codes = generate_huffman_codes(huff_tree)
            huffman_codes[name] = codes
            self.log_write(f"Generated {len(codes)} Huffman codes.")

            encoded_bitstream = bitarray()
            for symbol in channel_symbols:
                if symbol in codes: encoded_bitstream += codes[symbol]
                else: self.log_write(f"ERROR: Symbol '{symbol}' not in codes for {name}!")

            encoded_data_bits[name] = encoded_bitstream
            self.log_write(f"Encoded bitstream size: {len(encoded_bitstream)} bits")
            self.log_deindent()

        compressed_data_structure = {
            'original_image_shape': (self.image_height, self.image_width),
            'quality': self.quality, 'subsampling_mode': self.subsampling_mode,
            'grid_dims': grid_dims, 'original_channel_shapes': original_shapes,
            'huffman_codes': huffman_codes,
            'encoded_bits': encoded_data_bits
        }

        self.log_write("\n--- Saving Compressed Data Structure (Pickle) ---")
        try:
            with open(compressed_file, 'wb') as f: pickle.dump(compressed_data_structure, f)
            self.log_write(f"Saved data to {compressed_file}")
        except Exception as e:
            self.log_write(f"ERROR saving file: {e}"); print(f"Error saving file: {e}")

        self.log_write("\n--- Compression Finished ---")
        if self.log_file: self.log_file.close(); self.log_file = None


if __name__ == "__main__":
    img_height, img_width = 256, 256
    dummy_image_path = "imagenet-sample-images-resized/n01443537_goldfish.JPEG"

    img_options = ImageOptions(height=img_height, width=img_width)
    jpeg_options = JpegOptions(quality=15, subsampling_mode='4:2:0')

    print(f"Running JPEG compression with Entropy Coding...")
    compressor = JpegCompressorWithEntropy(img_options, jpeg_options)
    output_file = "compressed_jpeg_with_entropy.pkl"
    log_file = "compression_entropy_log.txt"

    compressor.compress_image(dummy_image_path, output_file, log_file)

    print(f"Compression finished. Check '{output_file}' and '{log_file}'.")
    import os
    try:
        original_size = os.path.getsize(dummy_image_path); compressed_size = os.path.getsize(output_file)
        print(f"\nOriginal: {original_size / 1024:.2f} KB, Compressed: {compressed_size / 1024:.2f} KB")
        if compressed_size > 0: print(f"Ratio: {original_size / compressed_size:.2f} : 1")
    except Exception as e: print(f"\nError comparing sizes: {e}")