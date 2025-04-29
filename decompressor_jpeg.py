# decompressor_jpeg.py

import numpy as np
from PIL import Image
import cv2
import scipy.fftpack
from math import ceil
import pickle
from typing import Any, Dict, Tuple, List, Optional, Union
import collections
from bitarray import bitarray
# from bitarray.util import ba2int, int2ba # Not needed for decode part

# --- Assumed Options/Base classes (can be simplified/removed if only decompressing) ---
class ImageOptions:
    def __init__(self, height: int, width: int): self.height = height; self.width = width
class JpegOptions:
    def __init__(self, quality: int = 75, subsampling_mode: str = '4:2:0'): self.quality = quality; self.subsampling_mode = subsampling_mode
class CompressorBase: # Minimal for logging
    def __init__(self): self._log_indent_level = 0; self.log_file = None
    def log_write(self, msg: str = ''):
        if self.log_file:
            indent = '  ' * self._log_indent_level
            try:
                self.log_file.write(indent + str(msg) + '\n')
                self.log_file.flush()
            except Exception as e:
                # Avoid crashing the main program if logging fails
                print(f"Logging Error: {e}")
    def log_indent(self): self._log_indent_level += 1
    def log_deindent(self): self._log_indent_level = max(0, self._log_indent_level - 1)

# --- Constants & Helpers (Need Q Tables, Zigzag, Huffman Node) ---
Q_LUM_ST = np.array([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],[14,13,16,24,40,57,69,56],[14,17,22,29,51,87,80,62],[18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],[49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]], dtype=np.int32)
Q_CHROM_ST = np.array([[17,18,24,47,99,99,99,99],[18,21,26,66,99,99,99,99],[24,26,56,99,99,99,99,99],[47,66,99,99,99,99,99,99],[99,99,99,99,99,99,99,99],[99,99,99,99,99,99,99,99],[99,99,99,99,99,99,99,99],[99,99,99,99,99,99,99,99]], dtype=np.int32)

zigzag_indices = np.array([[ 0, 1, 5, 6,14,15,27,28],[ 2, 4, 7,13,16,26,29,42],[ 3, 8,12,17,25,30,41,43],[ 9,11,18,24,31,40,44,53],[10,19,23,32,39,45,52,54],[20,22,33,38,46,51,55,60],[21,34,37,47,50,56,59,61],[35,36,48,49,57,58,62,63]])
zigzag_flat_indices = zigzag_indices.flatten(); zigzag_sort_indices = np.argsort(zigzag_flat_indices)
inverse_zigzag_indices = np.argsort(zigzag_sort_indices) # Precompute inverse zigzag indices
EOB = ('EOB',) # Define End-of-Block symbol as used in compressor
HuffmanNode = collections.namedtuple('HuffmanNode', ['symbol', 'freq', 'left', 'right']) # Needed for pickle loading if tree was saved


# --- The Decompressor Class ---
class JpegDecompressorFromPickle(CompressorBase):
    def __init__(self, compressed_file: str, log_file: Optional[str] = None):
        super().__init__()
        self.log_file = None
        # Ensure log file is opened *before* first log write
        if log_file:
            try: self.log_file = open(log_file, 'w')
            except IOError as e: print(f"Warning: Could not open log file {log_file}: {e}")

        self.log_write(f"--- Initializing Decompressor ---")
        self.log_write(f"Loading data from: {compressed_file}")
        try:
            with open(compressed_file, 'rb') as f:
                self.compressed_data = pickle.load(f)
            self.log_write("Pickle data loaded successfully.")
        except FileNotFoundError:
            self.log_write(f"ERROR: Compressed file not found at {compressed_file}")
            raise
        except pickle.UnpicklingError as e:
            self.log_write(f"ERROR: Failed to unpickle data from {compressed_file}: {e}")
            raise
        except Exception as e:
            self.log_write(f"ERROR: An unexpected error occurred loading {compressed_file}: {e}")
            raise

        try:
            self.original_image_shape = self.compressed_data['original_image_shape']
            self.quality = self.compressed_data['quality']
            self.subsampling_mode = self.compressed_data['subsampling_mode']
            self.grid_dims = self.compressed_data['grid_dims']
            self.original_channel_shapes = self.compressed_data['original_channel_shapes']
            self.huffman_codes = self.compressed_data['huffman_codes'] # Codes used for compression {symbol: bitarray}
            self.encoded_bits = self.compressed_data['encoded_bits']
            self.log_write(f"Metadata: Quality={self.quality}, Subsampling={self.subsampling_mode}, Shape={self.original_image_shape}")
        except KeyError as e:
            self.log_write(f"ERROR: Missing key in compressed data structure: {e}")
            raise ValueError(f"Pickle file is missing expected key: {e}")

        self.q_lum_scaled = self._scale_quantization_table(Q_LUM_ST, self.quality)
        self.q_chrom_scaled = self._scale_quantization_table(Q_CHROM_ST, self.quality)
        self.log_write("Re-calculated quantization tables.")

        self._build_inverse_huffman_maps() # Must happen after loading huffman_codes

    def _scale_quantization_table(self, q_table: np.ndarray, quality: int) -> np.ndarray:
        if quality < 1: quality = 1;
        if quality > 100: quality = 100
        scale = 5000 / quality if quality < 50 else 200 - 2 * quality
        scaled_table = np.floor((q_table * scale + 50) / 100)
        return np.clip(scaled_table, 1, 255).astype(np.float32) # Use float for dequant

    def _build_inverse_huffman_maps(self):
        self.inverse_huffman_maps = {} # Maps string codes back to symbols
        for name in ['y', 'cb', 'cr']:
            codes = self.huffman_codes.get(name, {})
            # Use string representation of bitarray as key
            self.inverse_huffman_maps[name] = {v.to01(): k for k, v in codes.items()}
            self.log_write(f"Built inverse Huffman map for {name.upper()} channel ({len(self.inverse_huffman_maps[name])} entries).")

    def _huffman_decode_stream(self, channel_name: str) -> List[Any]:
        self.log_write(f"Huffman decoding {channel_name.upper()} channel stream...")
        encoded_stream = self.encoded_bits.get(channel_name)
        inverse_map = self.inverse_huffman_maps.get(channel_name)

        if not encoded_stream or not inverse_map:
            self.log_write(f"Warning: No data or codes found for channel {channel_name}. Returning empty list.")
            return []

        decoded_symbols = []
        current_code_str = "" # Use a string to accumulate bits
        stream_pos = 0
        stream_len = len(encoded_stream)

        while stream_pos < stream_len:
            # Append '1' or '0' string based on the bit value
            current_code_str += '1' if encoded_stream[stream_pos] else '0'
            stream_pos += 1

            symbol = inverse_map.get(current_code_str) # Lookup using string key
            if symbol is not None:
                decoded_symbols.append(symbol)
                current_code_str = "" # Reset for next symbol

        # Check for leftover bits using the correct variable name
        if len(current_code_str) > 0:
             self.log_write(f"Warning: Trailing bits ('{current_code_str}') in stream for channel {channel_name} did not form a valid code.")

        # *** ADDED LOGGING: Check first few decoded symbols ***
        self.log_write(f"Decoded {len(decoded_symbols)} symbols. First 20: {decoded_symbols[:20]}")
        return decoded_symbols

    def _process_symbols_to_coeffs(self, symbols: List[Any], num_blocks: int) -> List[np.ndarray]:
        """Converts decoded symbols (DC diff, RLE AC) into zigzagged coefficient blocks."""
        self.log_write("Processing symbols into quantized zigzag coefficients...")
        coeffs_list = []
        previous_dc = 0
        symbol_idx = 0
        symbols_len = len(symbols)
        blocks_processed_count = 0 # To track blocks actually generated

        for block_num in range(num_blocks): # Iterate expected number of blocks
            block_log_prefix = f"  Block {block_num + 1}/{num_blocks}:" # Prefix for logs within this block
            coeffs = np.zeros(64, dtype=np.int32)

            # 1. Decode DC coefficient
            if symbol_idx >= symbols_len:
                 self.log_write(f"ERROR:{block_log_prefix} Ran out of symbols while expecting DC diff.")
                 coeffs_list.append(coeffs); continue # Add zero block and hope for the best? Or stop?

            current_symbol = symbols[symbol_idx]
            self.log_write(f"{block_log_prefix} Symbol {symbol_idx}: Processing DC symbol '{current_symbol}'")
            if isinstance(current_symbol, (int, np.integer)):
                dc_diff = current_symbol
                current_dc = dc_diff + previous_dc
                coeffs[0] = current_dc
                previous_dc = current_dc # Update predictor state
                self.log_write(f"{block_log_prefix}   DC Diff={dc_diff}, Prev DC={previous_dc-dc_diff}, Current DC={current_dc}. Set coeffs[0].")
                symbol_idx += 1
            else: # Error case
                self.log_write(f"ERROR:{block_log_prefix} Expected DC difference (int) at symbol index {symbol_idx}, got {type(current_symbol)} '{current_symbol}'. Skipping symbol.")
                symbol_idx += 1 # Skip the unexpected symbol
                # Keep coeffs[0] as 0 for this block, previous_dc unchanged

            # 2. Decode AC coefficients using RLE symbols
            ac_idx = 1 # Start filling from the second coefficient (index 1)
            self.log_write(f"{block_log_prefix} Starting AC decode at ac_idx={ac_idx}")
            while ac_idx < 64:
                if symbol_idx >= symbols_len:
                    self.log_write(f"ERROR:{block_log_prefix} Ran out of symbols while decoding AC (ac_idx={ac_idx}). Ending block early.")
                    break # Exit inner loop for this block

                symbol = symbols[symbol_idx]
                self.log_write(f"{block_log_prefix} Symbol {symbol_idx}: Processing AC symbol '{symbol}' (ac_idx={ac_idx})")
                symbol_idx += 1 # Consume the symbol

                if symbol == EOB:
                    self.log_write(f"{block_log_prefix}   EOB encountered. Ending AC for this block (ac_idx={ac_idx}).")
                    break # Stop processing AC for this block
                elif isinstance(symbol, tuple) and len(symbol) == 2:
                    run_length, value = symbol
                    if not (isinstance(run_length, (int, np.integer)) and isinstance(value, (int, np.integer))):
                         self.log_write(f"ERROR:{block_log_prefix} Invalid AC symbol format {symbol} at symbol index {symbol_idx-1}. Skipping.")
                         continue # Skip this malformed symbol

                    self.log_write(f"{block_log_prefix}   AC Symbol: Run={run_length}, Value={value}")

                    # Check boundary before advancing index for zeros
                    next_idx_after_zeros = ac_idx + run_length
                    if next_idx_after_zeros >= 64: # If placing zeros *reaches or exceeds* index 63
                         self.log_write(f"Warning:{block_log_prefix} Run length {run_length} (from ac_idx {ac_idx}) exceeds block boundary. Ending block early.")
                         break # Stop processing AC for this block

                    # Place Zeros (implicitly by advancing index)
                    self.log_write(f"{block_log_prefix}   Advancing ac_idx by {run_length} (zeros): {ac_idx} -> {next_idx_after_zeros}")
                    ac_idx = next_idx_after_zeros

                    # Place the non-zero value (boundary check implicitly done by ac_idx < 64 loop condition)
                    self.log_write(f"{block_log_prefix}   Placing Value={value} at ac_idx={ac_idx}")
                    coeffs[ac_idx] = value
                    ac_idx += 1 # Move to the next position *after* placing the value
                    self.log_write(f"{block_log_prefix}   Advanced ac_idx after value: -> {ac_idx}")

                else: # Unexpected symbol type
                    self.log_write(f"ERROR:{block_log_prefix} Unexpected symbol type {type(symbol)} '{symbol}' at symbol index {symbol_idx-1}. Expected EOB or (run, value). Skipping.")
                    continue # Skip this symbol

            # *** ADDED LOGGING: Log the first few complete coefficient blocks ***
            coeffs_list.append(coeffs)
            blocks_processed_count += 1
            if blocks_processed_count <= 5: # Log the first 5 blocks generated
                self.log_write(f"-> Finished Block {blocks_processed_count}. Final Coeffs: {coeffs.tolist()}")

        # Final checks after processing all expected blocks
        if blocks_processed_count != num_blocks:
             self.log_write(f"Warning: Processed {blocks_processed_count} blocks, but expected {num_blocks}.")
        if symbol_idx < symbols_len:
            self.log_write(f"Warning: {symbols_len - symbol_idx} symbols remaining after processing blocks. Remainder: {symbols[symbol_idx:symbol_idx+20]}")

        self.log_write(f"Reconstructed {len(coeffs_list)} coefficient blocks.")
        return coeffs_list

    def _inverse_zigzag_scan(self, zigzag_coeffs_1d: np.ndarray) -> np.ndarray:
        if len(zigzag_coeffs_1d) != 64: raise ValueError("Input must be 1D array of 64 coefficients.")
        block_flat = zigzag_coeffs_1d[inverse_zigzag_indices]
        return block_flat.reshape((8, 8))

    def _dequantize_block(self, quantized_block: np.ndarray, q_table: np.ndarray) -> np.ndarray:
        return quantized_block.astype(np.float32) * q_table # q_table is already float

    def _apply_idct_to_block(self, dequantized_dct_block: np.ndarray) -> np.ndarray:
        block_float = scipy.fftpack.idct(scipy.fftpack.idct(dequantized_dct_block.T, norm='ortho').T, norm='ortho')
        block_reconstructed = np.clip(np.round(block_float + 128.0), 0, 255)
        return block_reconstructed.astype(np.uint8)

    def _reassemble_from_blocks(self, blocks: List[np.ndarray], grid_h: int, grid_w: int) -> np.ndarray:
        if not blocks: self.log_write("Warning: No blocks to reassemble."); return np.array([])
        block_h, block_w = blocks[0].shape[:2]; dtype = blocks[0].dtype
        if block_h != 8 or block_w != 8: self.log_write(f"Warning: Block size is not 8x8 ({block_h}x{block_w}) during reassembly.")
        channel_h, channel_w = grid_h * block_h, grid_w * block_w
        reassembled = np.zeros((channel_h, channel_w), dtype=dtype)

        expected_blocks = grid_h * grid_w
        if len(blocks) != expected_blocks:
            self.log_write(f"Warning: Reassembling with {len(blocks)} blocks, but expected {expected_blocks}.")

        block_idx = 0
        for i in range(grid_h):
            for j in range(grid_w):
                if block_idx < len(blocks):
                    row_start, row_end = i * block_h, (i + 1) * block_h
                    col_start, col_end = j * block_w, (j + 1) * block_w
                    reassembled[row_start:row_end, col_start:col_end] = blocks[block_idx]
                    block_idx += 1
                else: break # Stop if we run out of blocks early
            if block_idx >= len(blocks): break
        return reassembled

    def _chroma_upsample(self, y_shape: Tuple[int, int], cb: np.ndarray, cr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.log_write(f"Upsampling chroma channels from Cb:{cb.shape}, Cr:{cr.shape} to match Y:{y_shape}...")
        target_h, target_w = y_shape
        # Use INTER_LINEAR for smoother results, matching common practice
        cb_upsampled = cv2.resize(cb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        cr_upsampled = cv2.resize(cr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        self.log_write(f"Upsampled shapes: Cb:{cb_upsampled.shape}, Cr:{cr_upsampled.shape}")
        return cb_upsampled, cr_upsampled

    def _ycrcb_to_rgb(self, y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
        self.log_write("Converting YCbCr back to RGB...")
        # Ensure channels have compatible shapes before merging
        if not (y.shape == cb.shape == cr.shape):
             self.log_write(f"ERROR: Channel shape mismatch before YCrCb->RGB conversion: Y={y.shape}, Cb={cb.shape}, Cr={cr.shape}")
             # Attempt to resize Cb/Cr to match Y as a fallback? Or raise error?
             target_shape = y.shape
             target_wh = (target_shape[1], target_shape[0]) # (width, height) for cv2.resize
             self.log_write(f"Attempting to resize Cb/Cr to {target_shape}...")
             if cb.shape != target_shape: cb = cv2.resize(cb, target_wh, interpolation=cv2.INTER_LINEAR)
             if cr.shape != target_shape: cr = cv2.resize(cr, target_wh, interpolation=cv2.INTER_LINEAR)

        # OpenCV expects Y Cr Cb order for merging/conversion
        ycrcb_image = cv2.merge([y, cr, cb])
        image_rgb = cv2.cvtColor(ycrcb_image, cv2.COLOR_YCrCb2RGB)
        self.log_write("Color conversion done.")
        return image_rgb

    # --- Main Decompression Method ---
    def decompress_to_rgb_array(self) -> Optional[np.ndarray]:
        self.log_write("\n--- Starting Decompression Process ---")
        reconstructed_channels = {}

        for name, q_table in [('y', self.q_lum_scaled), ('cb', self.q_chrom_scaled), ('cr', self.q_chrom_scaled)]:
            self.log_write(f"\nDecompressing {name.upper()} Channel...")
            self.log_indent()

            decoded_symbols = self._huffman_decode_stream(name)
            if not decoded_symbols:
                 self.log_write(f"Warning: No symbols decoded for channel {name}. Skipping channel.");
                 reconstructed_channels[name] = None; self.log_deindent(); continue

            grid_h, grid_w = self.grid_dims[name]
            num_blocks = grid_h * grid_w
            zigzagged_coeffs_list = self._process_symbols_to_coeffs(decoded_symbols, num_blocks)
            if len(zigzagged_coeffs_list) != num_blocks:
                 self.log_write(f"ERROR: Coeff block count mismatch ({len(zigzagged_coeffs_list)} vs {num_blocks}). Skipping channel.")
                 reconstructed_channels[name] = None; self.log_deindent(); continue

            reconstructed_blocks_list = []
            for i, zigzag_coeffs in enumerate(zigzagged_coeffs_list):
                 quantized_block = self._inverse_zigzag_scan(zigzag_coeffs)
                 dequantized_block = self._dequantize_block(quantized_block, q_table)
                 reconstructed_block = self._apply_idct_to_block(dequantized_block)
                 reconstructed_blocks_list.append(reconstructed_block)

            # *** ADDED LOGGING: Check first reconstructed pixel block ***
            if reconstructed_blocks_list:
                 self.log_write(f"First reconstructed pixel block (after IDCT):\n{reconstructed_blocks_list[0]}")

            padded_channel = self._reassemble_from_blocks(reconstructed_blocks_list, grid_h, grid_w)
            self.log_write(f"Reassembled padded channel shape: {padded_channel.shape}")

            orig_h, orig_w = self.original_channel_shapes.get(name, (0,0)) # Get original shape pre-padding
            if orig_h == 0 or orig_w == 0:
                 self.log_write(f"ERROR: Could not find original shape for channel {name}. Cannot unpad.")
                 reconstructed_channels[name] = None; self.log_deindent(); continue

            unpadded_channel = padded_channel[:orig_h, :orig_w]
            reconstructed_channels[name] = unpadded_channel
            self.log_write(f"Unpadded channel {name.upper()} to original shape: {unpadded_channel.shape}")
            self.log_deindent()

        # Check if any value *is* None, avoiding NumPy comparison ambiguity
        if any(value is None for value in reconstructed_channels.values()):
             self.log_write("ERROR: Failed to reconstruct one or more channels. Aborting final assembly.")
             # Close log file if open
             if self.log_file: self.log_file.close(); self.log_file = None
             return None

        self.log_write("\n--- Final Assembly ---")
        y_unpadded = reconstructed_channels['y']
        cb_unpadded = reconstructed_channels['cb']
        cr_unpadded = reconstructed_channels['cr']

        target_h, target_w = self.original_image_shape
        if cb_unpadded.shape != (target_h, target_w) or cr_unpadded.shape != (target_h, target_w):
             cb_upsampled, cr_upsampled = self._chroma_upsample((target_h, target_w), cb_unpadded, cr_unpadded)
        else:
             self.log_write("Chroma channels already at full resolution. Skipping upsampling.")
             cb_upsampled, cr_upsampled = cb_unpadded, cr_unpadded

        y_final = y_unpadded[:target_h, :target_w]
        if y_final.shape != (target_h, target_w):
             self.log_write(f"Warning: Final Y channel shape {y_final.shape} doesn't match target {target_h, target_w}. Assuming it's correct.")
             # Optionally resize: y_final = cv2.resize(y_final, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        reconstructed_rgb = self._ycrcb_to_rgb(y_final, cb_upsampled, cr_upsampled)

        self.log_write("\n--- Decompression Finished Successfully ---")
        if self.log_file:
            self.log_file.close()
            self.log_file = None

        return reconstructed_rgb


# --- Example Usage ---
if __name__ == "__main__":
    compressed_pickle_file = "compressed_jpeg_with_entropy.pkl"
    decompression_log_file = "decompression_log.txt"
    output_image_file = "reconstructed_image.png" # Save as PNG

    try:
        print(f"--- Decompressing {compressed_pickle_file} ---")
        decompressor = JpegDecompressorFromPickle(compressed_pickle_file, decompression_log_file)
        reconstructed_array = decompressor.decompress_to_rgb_array()

        if reconstructed_array is not None:
            print(f"Decompression logic completed. Reconstructed array shape: {reconstructed_array.shape}")
            # Check if array is uniform (potential sign of issue)
            if np.all(reconstructed_array == reconstructed_array[0,0]):
                print("Warning: Reconstructed array appears to be uniform (blank/single color). Check logs for details.")
            elif np.max(reconstructed_array) - np.min(reconstructed_array) < 10: # Low contrast check
                print("Warning: Reconstructed array has very low contrast. Check logs.")

            try:
                img_out = Image.fromarray(reconstructed_array)
                img_out.save(output_image_file)
                img_out.save("reconstructed_image_again.jpg", quality=50)
                print(f"Successfully saved reconstructed image to {output_image_file}")
                # img_out.show() # Optional display
            except Exception as e:
                print(f"Error saving or showing the reconstructed image: {e}")
        else:
            print("Decompression failed during processing. Check logs.")

    except FileNotFoundError:
        print(f"ERROR: Input pickle file '{compressed_pickle_file}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred during decompression: {e}")
        import traceback
        traceback.print_exc()