# compressor.py

import numpy as np
from PIL import Image
from bitarray import bitarray
from bitarray.util import ba2int, int2ba # Assuming int2ba is in bitarray.util or helper
# --- Make sure these imports work based on your project structure ---
from helper import * # Assumes options classes, float2ba etc. are here
from bloom import BloomFilter # Assumes BloomFilter class is here
from options import * # Assumes options classes are defined here too (redundant OK)
from compressor_base import CompressorBase # Import the fixed base class
# --- End Project Structure Imports ---
from typing import Any, Callable, Union, Dict, Optional, List, Tuple # Ensure Optional etc. are imported
import collections
import heapq
import pickle # Keep import if you plan alternate saving methods

# --- Huffman Coding Helper ---
HuffmanNode = collections.namedtuple('HuffmanNode', ['symbol', 'freq', 'left', 'right'])
HuffmanNode.__lt__ = lambda x, y: x.freq < y.freq

def build_huffman_tree(frequencies: collections.Counter) -> Optional[HuffmanNode]:
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

def generate_huffman_codes(tree: Optional[HuffmanNode]) -> Dict[Any, bitarray]:
    codes = {}
    if tree is None: return codes
    def traverse(node, current_code):
        if node.symbol is not None: # Leaf node
            codes[node.symbol] = current_code if current_code else bitarray('0')
            return
        if node.left: traverse(node.left, current_code + bitarray('0'))
        if node.right: traverse(node.right, current_code + bitarray('1'))
    traverse(tree, bitarray()); return codes

def str2ba(s: str) -> bitarray: # Helper to convert '0101' string to bitarray
    return bitarray(s)

class Compressor(CompressorBase):
    def __init__(self,
                 image_options: ImageOptions,
                 quantization_options: Optional[QuantizationOptions],
                 segment_options: Optional[SegmentOptions],
                 linear_regression_options: Optional[LinearRegressionOptions],
                 delta_encoding_options: Optional[DeltaEncodingOptions],
                 bloom_filter_options: Optional[BloomFilterOptions],
                ):
        # Call base class init FIRST to ensure self.log_file exists (as None)
        super().__init__()

        # --- VALIDATE AND SET OPTIONS ---
        # Image settings
        image_options.validate()
        self.image_height = image_options.height
        self.image_width = image_options.width

        # Quantization settings
        if quantization_options:
            quantization_options.validate()
            self.quantization_bits = quantization_options.quantization_bits
        else:
            self.quantization_bits = 8 # Default if None

        # Segmenting settings
        if segment_options:
            # segment_options.validate(self.image_height, self.image_width) # Assuming validate exists
            self.segment_height = segment_options.height
            self.segment_width = segment_options.width
        else:
            self.segment_height = self.image_height # Default: no segments
            self.segment_width = self.image_width

        # Linear regression settings
        if linear_regression_options:
            # linear_regression_options.validate() # Assuming validate exists
            self.do_linear_regression = True
        else:
            self.do_linear_regression = False

        # Delta encoding settings
        if delta_encoding_options:
            # delta_encoding_options.validate() # Assuming validate exists
            self.do_delta_encoding = True
            self.anchor_frequency = delta_encoding_options.anchor_frequency
            self.smear_bits = delta_encoding_options.smear_bits
            if self.anchor_frequency <= 0:
                 raise ValueError("Anchor frequency must be positive.")
            if not (1 <= self.smear_bits <= 8): # Example range check
                 raise ValueError("Smear bits must be between 1 and 8.")
        else:
            self.do_delta_encoding = False
            self.anchor_frequency = 1 # Avoid division by zero later if delta off
            self.smear_bits = 8       # Default doesn't matter if delta off

        # Bloom filter settings
        if bloom_filter_options:
            # bloom_filter_options.validate() # Assuming validate exists
            self.do_bloom_filter = True
            if not (0 < bloom_filter_options.fpp < 1):
                 raise ValueError("Bloom filter FPP must be between 0 and 1.")
            ln_fpp = np.log(bloom_filter_options.fpp)
            ln2 = np.log(2)
            self.bloom_bits_per_element: float = -ln_fpp / ln2**2
            self.bloom_hash_count: int = round(-ln_fpp / ln2)
            if self.bloom_hash_count <= 0 : self.bloom_hash_count = 1 # Ensure at least one hash
        else:
            self.do_bloom_filter = False
            # Set defaults even if not used, avoid potential errors later
            self.bloom_bits_per_element = 0
            self.bloom_hash_count = 0


        # Set parameters derived from the ones above
        # This call is now safe because self.log_file exists (as None)
        self.set_dependent_parameters()

    def set_dependent_parameters(self):
        """Calculate parameters derived from options. Safe to call log_write."""
        self.segment_size = self.segment_height * self.segment_width
        if self.quantization_bits <= 0 or self.quantization_bits > 8:
            raise ValueError("Quantization bits must be between 1 and 8.")
        self.quantization_factor = 2**(8 - self.quantization_bits)
        self.quantization_max = 2**self.quantization_bits - 1

        if self.do_delta_encoding:
            # Bits needed for anchors (storing quantized values)
            self.bits_per_anchor_val = self.quantization_bits
            self.is_signed_anchor = False # Quantized values 0..max are unsigned

            # Bits/range needed for smeared values (smear_bits includes sign)
            self.smear_max = 2**(self.smear_bits - 1) - 1 # Max positive smeared delta
            self.bits_per_main_val = self.smear_bits # Use smear_bits for non-anchor deltas
            self.is_signed_main_val = True # Deltas/Residuals can be negative
        else:
            # If no delta encoding, main values are the quantized pixel values
            self.bits_per_main_val = self.quantization_bits
            self.is_signed_main_val = False # Quantized values 0..max are unsigned

        # Log derived parameters (will only write if log file is set later)
        self.log_write("--- Derived Parameters ---")
        self.log_write(f"Segment Size: {self.segment_size}")
        self.log_write(f"Quantization Factor: {self.quantization_factor}")
        self.log_write(f"Quantization Max Value: {self.quantization_max}")
        if self.do_delta_encoding:
            self.log_write(f"Anchor Bits: {self.bits_per_anchor_val}, Signed: {self.is_signed_anchor}")
            self.log_write(f"Smear Bits: {self.smear_bits}, Smear Max Value: +/- {self.smear_max}")
        self.log_write(f"Main Value Bits (Huffman/Bloom): {self.bits_per_main_val}, Signed: {self.is_signed_main_val}")
        self.log_write(f"Bloom Filter Enabled: {self.do_bloom_filter}")
        if self.do_bloom_filter:
            self.log_write(f" Bloom Bits/Elem: {self.bloom_bits_per_element:.2f}, Hash Count: {self.bloom_hash_count}")
        self.log_write("--------------------------")


    def compress_image(self, image_file: str, compressed_file: str, log_file: Optional[str] = None):
        """Compresses the image using the configured pipeline."""
        # --- Set up logging using the base class method ---
        # This handles opening the file and setting self.log_file
        self.set_log_file(log_file)

        self.log_write(f"--- Starting Compression: {image_file} -> {compressed_file} ---")
        try:
            # Load and validate image
            img = Image.open(image_file).convert('RGB') # Ensure RGB
            arr = np.array(img)
            if not (arr.shape[0] == self.image_height and arr.shape[1] == self.image_width):
                 raise ValueError(f"Input image size {arr.shape[:2]} != options {self.image_height}x{self.image_width}")
            if arr.ndim != 3 or arr.shape[2] != 3: raise ValueError("Invalid image shape.")
            if not ((0 <= arr).all() and (arr <= 255).all()): self.log_write("Warning: Image pixel values outside [0, 255].") # Allow but warn
            self.log_write(f"Loaded image shape: {arr.shape}, dtype: {arr.dtype}")

            # Initialize bitarray for output
            self.compressed_bitarray = bitarray()

            # --- Write Metadata (Fixed Length) ---
            self.log_write("--- Writing Metadata ---")
            self.write(self.image_height, int2ba, 16, msg='Image height: ')
            self.write(self.image_width, int2ba, 16, msg='Image width: ')
            self.write(self.quantization_bits, int2ba, 4, msg='Quantization bits: ')
            self.write(self.segment_height, int2ba, 16, msg='Segment height: ')
            self.write(self.segment_width, int2ba, 16, msg='Segment width: ')
            self.write(int(self.do_linear_regression), int2ba, 1, msg='Linear regression?: ')
            self.write(int(self.do_delta_encoding), int2ba, 1, msg='Delta encoding?: ')
            self.write(int(self.do_bloom_filter), int2ba, 1, msg='Bloom filter?: ')
            if self.do_delta_encoding:
                self.write(self.anchor_frequency, int2ba, 16, msg='Anchor frequency: ') # 16 bits for anchor freq?
                self.write(self.smear_bits, int2ba, 4, msg='Smear bits: ')
            if self.do_bloom_filter:
                if not (0 <= self.bloom_hash_count <= 255): raise ValueError("Bloom hash count out of range for 8 bits")
                self.write(self.bloom_hash_count, int2ba, 8, msg='Bloom hash count: ')
            self.log_write("--- End Metadata ---")


            # --- Prepare for Segmentation ---
            if self.do_delta_encoding:
                if self.segment_size <= 0: raise ValueError("Segment size must be positive for delta encoding.")
                anchor_mask = np.zeros(self.segment_size, dtype=bool)
                anchor_mask[::self.anchor_frequency] = True # Correct assignment

            self.log_write("Quantizing image...")
            arr = self.quantize(arr)
            self.log_write("Segmenting image...")
            segments = self.segment(arr)
            self.log_write(f"Processing {len(segments)} segments...")

            # --- Process Segments ---
            for segment_idx, segment in enumerate(segments):
                self.log_write(f'\n--- SEGMENT {segment_idx} ---')
                self.log_indent()

                red = segment[:, 0].copy(); green = segment[:, 1].copy(); blue = segment[:, 2].copy()

                # Linear Regression (Fixed Length)
                if self.do_linear_regression:
                    self.log_write("Applying Linear Regression...")
                    try:
                        green, green_beta0, green_beta1 = self.linear_regression(red, green)
                        blue, blue_beta0, blue_beta1 = self.linear_regression(red, blue)
                        self.write(green_beta0, float2ba, 32, True, msg='Green beta0: ')
                        self.write(green_beta1, float2ba, 32, True, msg='Green beta1: ')
                        self.write(blue_beta0, float2ba, 32, True, msg='Blue beta0: ')
                        self.write(blue_beta1, float2ba, 32, True, msg='Blue beta1: ')
                    except ZeroDivisionError: # Catch specific error from regression
                         self.log_write("Warning: Skipping linear regression (predictor constant). Writing default coefficients.")
                         self.write(0.0, float2ba, 32, True, msg='Green beta0 (default): ')
                         self.write(0.0, float2ba, 32, True, msg='Green beta1 (default): ')
                         self.write(0.0, float2ba, 32, True, msg='Blue beta0 (default): ')
                         self.write(0.0, float2ba, 32, True, msg='Blue beta1 (default): ')


                # Process Channels within Segment
                for channel_data, channel_name in zip((red, green, blue), ('RED', 'GREEN', 'BLUE')):
                    self.log_write(f'\n-- {channel_name} CHANNEL --')
                    self.log_indent()
                    processed_data = channel_data

                    # Delta Encoding & Smearing (if applicable)
                    if self.do_delta_encoding:
                        self.log_write("Applying Delta Encoding & Smearing...")
                        delta_arr = self.delta_encoding(processed_data)
                        smeared_arr = self.smear(delta_arr)

                        # Write Anchors (Fixed Length)
                        anchors = smeared_arr[anchor_mask]
                        self.log_write(f"Writing {len(anchors)} anchors...")
                        self.write(anchors, intarr2ba, self.bits_per_anchor_val, self.is_signed_anchor, msg=f'Anchors ({len(anchors)} vals):')
                        sparse_arr = smeared_arr[~anchor_mask]
                        self.log_write(f"Proceeding with {len(sparse_arr)} non-anchor values.")
                    else:
                        sparse_arr = processed_data
                        self.log_write(f"Proceeding with {len(sparse_arr)} direct values (no delta).")


                    # --- Bloom Filter Path OR Huffman Path ---
                    if self.do_bloom_filter:
                        self.log_write("Applying Bloom Filter Logic...")
                        self.log_write("WARNING: Bloom filter path is inefficient for compression storage.")
                        if sparse_arr.size == 0:
                             self.log_write(" Empty sparse array. Writing mode=0, count=0.")
                             self.write(0, int2ba, self.bits_per_main_val, self.is_signed_main_val, msg='Mode (Empty): ')
                             self.write(0, int2ba, 16, msg='Num Bloom Filters (Empty): ') # 16 bits for count
                        else:
                             bloom_filters, mode = self.bloom_filter(sparse_arr)
                             self.log_write(f"Mode value: {mode}, Number of filters: {len(bloom_filters)}")
                             self.write(mode, int2ba, self.bits_per_main_val, self.is_signed_main_val, msg='Mode: ')
                             count_bits = 16 # Bits for filter count
                             self.write(len(bloom_filters), int2ba, count_bits, msg=f'Num Bloom Filters ({count_bits} bits): ')
                             for num, bloom_filter in bloom_filters.items():
                                 self.log_write(f' Writing Bloom Filter for value: {num}')
                                 self.log_indent()
                                 self.write(num, int2ba, self.bits_per_main_val, self.is_signed_main_val, msg=' Filter Value: ')
                                 filter_size = len(bloom_filter.bit_array)
                                 self.write(filter_size, int2ba, 32, msg=' Filter Size (bits): ') # 32 bits for size
                                 self.append_bitarray(bloom_filter.bit_array, msg=f' Filter Data ({filter_size} bits): ')
                                 self.log_deindent()
                    else:
                        # Huffman Coding Path
                        self.log_write("Applying Huffman Coding...")
                        if sparse_arr.size == 0:
                            self.log_write(" Empty sparse array, writing 0 codes count.")
                            self.write(0, int2ba, 16, msg='Code Count (Empty): ') # 16 bits for code count
                        else:
                            try:
                                symbols_for_freq = [int(x) for x in sparse_arr] # Use standard ints as keys
                                freqs = collections.Counter(symbols_for_freq)
                                self.log_write(f" Found {len(freqs)} unique symbols. Top 5: {freqs.most_common(5)}")

                                tree = build_huffman_tree(freqs)
                                codes = generate_huffman_codes(tree) # {symbol: bitarray}

                                num_codes = len(codes)
                                self.log_write(f" Storing Huffman table with {num_codes} entries...")
                                self.write(num_codes, int2ba, 16, msg=f' Code Count ({num_codes}): ')
                                self.log_indent()
                                for symbol, code_ba in codes.items():
                                    # Write Symbol (using bits/signedness determined earlier)
                                    self.write(symbol, int2ba, self.bits_per_main_val, self.is_signed_main_val, msg=f'  Symbol {symbol}: ')
                                    # Write Code string representation
                                    code_str = code_ba.to01()
                                    code_len = len(code_str)
                                    if code_len > 255: self.log_write(f"Warning: Huffman code len {code_len} > 255!")
                                    self.write(code_len, int2ba, 8, msg=f'Code Len {code_len}: ') # 8 bits for code length
                                    self.append_bitarray(str2ba(code_str), msg=f'Code "{code_str}": ')
                                self.log_deindent()

                                self.log_write(f" Encoding {len(sparse_arr)} data symbols...")
                                encoded_data = bitarray()
                                symbols_encoded_count = 0
                                for val in sparse_arr:
                                    symbol_key = int(val) # Use standard int for lookup
                                    if symbol_key in codes:
                                        encoded_data += codes[symbol_key]
                                        symbols_encoded_count +=1
                                    else:
                                        self.log_write(f"ERROR: Symbol '{symbol_key}' not in Huffman codes! Skipping value.")
                                self.log_write(f" Encoded data length: {len(encoded_data)} bits ({symbols_encoded_count}/{len(sparse_arr)} symbols).")
                                self.append_bitarray(encoded_data, msg="Encoded Data:")

                            except Exception as e:
                                # Log error and write 0 codes count as a fallback
                                self.log_write(f"ERROR during Huffman processing for channel {channel_name}, segment {segment_idx}: {e}")
                                self.write(0, int2ba, 16, msg="Code Count (Error): ")

                    self.log_deindent() # Channel indent
                self.log_deindent() # Segment indent

            # --- Write the compressed bitarray to file ---
            self.log_write(f"\n--- Writing final bitarray ({len(self.compressed_bitarray)} bits) to {compressed_file} ---")
            with open(compressed_file, 'wb') as f:
                self.compressed_bitarray.tofile(f)
            self.log_write("File write successful.")

        except Exception as e:
            self.log_write(f"\n!!! CRITICAL ERROR during compression: {e} !!!")
            import traceback
            traceback.print_exc(file=self.log_file) # Log traceback if possible
            print(f"Compression failed due to error: {e}") # Also print to console


    # Modified write function using base class logging
    def write(self,
              val: Any,
              func: Callable | None = None, # e.g., int2ba, float2ba
              length: int | None = None,
              signed: bool = False,
              msg: Any = None):
        """Converts value using func and appends to bitarray. Handles basic logging."""
        bits_to_append = bitarray() # Default to empty
        try:
            if func:
                bits_to_append = func(val, length, signed)
            elif isinstance(val, bitarray):
                bits_to_append = val
            else:
                # Fallback if no function and not bitarray: try converting val to string and then bitarray? Unreliable.
                self.log_write(f"ERROR in write: No function provided and value is not bitarray (type: {type(val)}). Cannot append.")
                # Or raise ValueError("Cannot write non-bitarray value without conversion function")

            if not isinstance(bits_to_append, bitarray):
                self.log_write(f"ERROR in write: Conversion function did not return bitarray (returned {type(bits_to_append)}). Appending empty.")
                bits_to_append = bitarray() # Ensure we only append bitarrays

            # Log before appending
            if msg is not None:
                log_val_str = str(val)
                # Shorten logging for large data types
                if isinstance(val, np.ndarray) and val.size > 10: log_val_str = f"ndarray shape {val.shape} dtype {val.dtype}"
                elif isinstance(val, bitarray) and len(val) > 80: log_val_str = f"bitarray len {len(val)}"
                elif isinstance(val, list) and len(val) > 10: log_val_str = f"list len {len(val)}"
                self.log_write(str(msg) + log_val_str + f" -> {len(bits_to_append)} bits")

            self.compressed_bitarray += bits_to_append

        except Exception as e:
             self.log_write(f"ERROR during write operation for msg='{msg}': {e}")
             # Decide if error is fatal or if we can continue (e.g., append empty bits)


    # New method specifically for appending bitarrays (used by Huffman/Bloom data)
    def append_bitarray(self, ba: bitarray, msg: Any = None):
         """Appends a pre-formed bitarray and logs."""
         if not isinstance(ba, bitarray):
            self.log_write(f"ERROR in append_bitarray: Input is not a bitarray (type: {type(ba)}). Skipping.")
            return
         # Log before appending
         if msg is not None:
            log_val_str = f"bitarray len {len(ba)}"
            if len(ba) <= 80: log_val_str += f" ({ba.to01()})" # Log content if short
            self.log_write(str(msg) + log_val_str)
         self.compressed_bitarray += ba


    # --- Keep your existing methods: quantize, segment, linear_regression, delta_encoding, smear, bloom_filter ---
    # Ensure these methods use self.log_write for logging if needed
    # Make sure linear_regression handles potential division by zero if x is constant
    def quantize(self, arr: np.ndarray) -> np.ndarray:
        self.log_write(f" Quantizing array shape {arr.shape} with factor {self.quantization_factor}")
        return arr // self.quantization_factor

    def segment(self, arr: np.ndarray) -> list[np.ndarray]:
        self.log_write(f" Segmenting array shape {arr.shape} into {self.segment_height}x{self.segment_width} segments")
        image_height, image_width, image_channels = arr.shape
        segments = []
        for row_start in range(0, image_height, self.segment_height):
            for col_start in range(0, image_width, self.segment_width):
                row_end = min(row_start + self.segment_height, image_height)
                col_end = min(col_start + self.segment_width, image_width)
                segment_arr = arr[row_start:row_end, col_start:col_end, :]
                if segment_arr.size > 0:
                    flattened_segment_arr = segment_arr.reshape(-1, image_channels)
                    segments.append(flattened_segment_arr)
        return segments

    def linear_regression(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
        x_f = x.astype(float); y_f = y.astype(float) # Use different names for float versions
        xbar = x_f.mean(); ybar = y_f.mean()
        ss_xx = ((x_f-xbar)**2).sum()
        if np.isclose(ss_xx, 0): # Safer check for near-zero variance
            beta1 = 0.0
            # Log handled in calling function now
        else:
            beta1 = ((x_f-xbar)*(y_f-ybar)).sum() / ss_xx
        beta0 = ybar - beta1 * xbar
        # Calculate prediction based on original quantized 'x' values if needed?
        # Or predict based on x_f and then clip/quantize? Let's use x_f.
        yhat = beta0 + beta1 * x_f
        # Clip predictions to the valid *quantized* range
        yhat = yhat.clip(0, self.quantization_max).astype(int)
        # Residuals calculated from original *quantized* y and *clipped integer* yhat
        # Ensure y is treated as int here if it wasn't already
        res = (y.astype(int) - yhat).astype(int)
        return res, beta0, beta1

    def delta_encoding(self, arr: np.ndarray) -> np.ndarray:
        enc = np.zeros_like(arr, dtype=int)
        if arr.size > 1: # Need at least 2 elements for diff
            diffs = np.diff(arr.astype(int))
            enc[1:] = diffs
        # Place original anchor values (handles anchor_freq=1 correctly)
        enc[::self.anchor_frequency] = arr[::self.anchor_frequency]
        return enc

    def smear(self, diff_arr: np.ndarray) -> np.ndarray:
         smeared_arr = np.zeros_like(diff_arr, dtype=int)
         carry = 0
         # Use smear_max calculated in set_dependent_parameters
         smear_limit = getattr(self, 'smear_max', 3) # Default if not set (e.g., smear_bits=4)

         for i in range(diff_arr.shape[0]):
             is_anchor = (i > 0) and (i % self.anchor_frequency == 0) # Don't reset carry at index 0 unless it's an anchor conceptually
             # More precise anchor check:
             # is_anchor = (i % self.anchor_frequency == 0) # Original logic

             current_val = diff_arr[i]

             # Anchors store actual values, not deltas - reset carry at anchor
             if i % self.anchor_frequency == 0:
                 carry = 0
                 smeared_arr[i] = current_val # Anchors are not smeared deltas
             else: # It's a difference value, apply smearing
                 carried_diff = current_val + carry
                 clipped_diff = np.clip(carried_diff, -smear_limit, smear_limit)
                 smeared_arr[i] = clipped_diff
                 carry = carried_diff - clipped_diff

         return smeared_arr

    def bloom_filter(self, sparse_arr: np.ndarray) -> tuple[dict[int, BloomFilter], int]:
        # Use parameters calculated in set_dependent_parameters
        bits_per_elem = getattr(self, 'bloom_bits_per_element', 10)
        hash_count = getattr(self, 'bloom_hash_count', 7)

        num_to_indices: dict[int, list[int]] = {}
        if sparse_arr.size == 0: return {}, 0 # Handle empty

        for i, num in enumerate(sparse_arr):
            num_int = int(num) # Use standard int for dict keys
            if num_int not in num_to_indices: num_to_indices[num_int] = []
            num_to_indices[num_int].append(i)

        if not num_to_indices: return {}, 0

        max_num_count = -1; mode_val = list(num_to_indices.keys())[0] # Default mode
        for num, indices in num_to_indices.items():
            if len(indices) > max_num_count:
                max_num_count = len(indices)
                mode_val = num # Found a more frequent mode

        bloom_filters: dict[int, BloomFilter] = {}
        for num, indices in num_to_indices.items():
            if num == mode_val: continue # Skip mode
            size = max(1, round(len(indices) * bits_per_elem))
            final_hash_count = max(1, hash_count) # Ensure > 0
            try:
                bf = BloomFilter(size, final_hash_count) # Ensure class takes size, count
                for index in indices: bf.add(index)
                bloom_filters[num] = bf
            except Exception as e:
                self.log_write(f"ERROR creating Bloom filter for {num} (size={size}, count={final_hash_count}): {e}")
        return bloom_filters, mode_val


# --- Example Usage ---
if __name__ == "__main__":
    # Configure options - USE HUFFMAN (Bloom Filter = None)
    compressor_huffman = Compressor(
        image_options=ImageOptions(height=256, width=256),
        quantization_options=QuantizationOptions(quantization_bits=4), # More aggressive quantization
        segment_options=SegmentOptions(height=64, width=64),
        linear_regression_options=LinearRegressionOptions(),
        delta_encoding_options=DeltaEncodingOptions(anchor_frequency=16, smear_bits=4),
        bloom_filter_options=None # *** USE HUFFMAN ***
    )

    input_image = 'imagenet-sample-images-resized/n01443537_goldfish.JPEG'
    output_file_huffman = 'compressed_custom_huffman.bin'
    log_file_huffman = 'compressor_custom_huffman_log.txt'

    print(f"--- Running Custom Compressor with Huffman ---")
    print(f"Input: {input_image}")
    print(f"Output: {output_file_huffman}")
    print(f"Log: {log_file_huffman}")
    # Run compression within a try block for better error reporting
    try:
        compressor_huffman.compress_image(input_image, output_file_huffman)
        print(f"--- Compression finished ---")
    except Exception as main_e:
        print(f"--- Compression FAILED ---")
        print(f"Error: {main_e}")
        import traceback
        traceback.print_exc() # Print full traceback to console

    # Optional: Compare file size
    import os
    if os.path.exists(output_file_huffman):
        try:
            original_size = os.path.getsize(input_image)
            compressed_size = os.path.getsize(output_file_huffman)
            print(f"\nOriginal size: {original_size / 1024:.2f} KB")
            print(f"Compressed size: {compressed_size / 1024:.2f} KB")
            if compressed_size > 0 and original_size > 0 :
                 print(f"Compression ratio: {original_size / compressed_size:.2f} : 1")
            if compressed_size > original_size:
                print("Warning: Compressed file is larger than original!")
        except FileNotFoundError:
            print("\nCould not compare file sizes (file not found).")
        except Exception as e:
            print(f"\nError comparing file sizes: {e}")
    else:
        print(f"\nCompressed file '{output_file_huffman}' not created.")