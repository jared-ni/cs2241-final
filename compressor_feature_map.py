import numpy as np
from bitarray import bitarray
from helper import *
from compressor_base_feature_map import FeatureMapCompressorBase
from bloom import BloomFilter
from count_min_sketch import CountMinSketch
from options_feature_map import *
import heapq
from collections import Counter
from bloomier_filter import BloomierFilter
from typing import Any, Callable

class FeatureMapCompressor(FeatureMapCompressorBase):
    def __init__(self,
                 feature_map_options: FeatureMapOptions,
                 quantization_options: QuantizationOptions,
                 count_min_options: CountMinOptions | None = None,
                 bloomier_options: BloomierOptions | None = None,
                 huffman_options: HuffmanOptions | None = None,
        ):
        super().__init__()

        # Feature map settings
        feature_map_options.validate()
        self.feature_map_shape = feature_map_options.shape

        # Quantization settings
        quantization_options.validate()
        self.quantization_bits = quantization_options.quantization_bits
        
        # Count Min sketch with Bloom filter settings
        if count_min_options:
            count_min_options.validate()
            self.do_count_min = True
            if count_min_options.bloom_fpp is None:
                self.bloom_hash_count = 0
            else:
                ln_fpp = np.log(count_min_options.bloom_fpp)
                ln2 = np.log(2)
                self.bloom_bits_per_element: float = -ln_fpp / ln2**2
                self.bloom_hash_count: int = round(-ln_fpp / ln2)
            self.cm_width = int(np.ceil(np.e / count_min_options.cm_epsilon))
            self.cm_depth = int(np.ceil(np.log(1 / count_min_options.cm_delta)))
        else:
            self.do_count_min = False

        # Bloomier filter settings
        if bloomier_options:
            bloomier_options.validate()
            self.do_bloomier = True
            self.bloomier_second_table = bloomier_options.second_table
            self.bloomier_fpp = bloomier_options.fpp
            self.bloomier_slots_per_key = bloomier_options.slots_per_key
            self.bloomier_hash_count = bloomier_options.hash_count
        else:
            self.do_bloomier = False

        # Cannot do both Count Min and Bloomier
        assert not (self.do_count_min and self.do_bloomier)

        # Huffman settings
        if huffman_options:
            huffman_options.validate()
            self.do_huffman = True
            self.huffman_symbol_size = huffman_options.symbol_size
        else:
            self.do_huffman = False

    def compress_feature_map(self, feature_map_file: str, compressed_file: str, log_file: str | None = None):
        feature_map: np.ndarray = np.load(feature_map_file)
        assert feature_map.shape == self.feature_map_shape
        feature_map = feature_map.flatten()
        self.feature_map_max_val = feature_map.max()

        self.compressed_bitarray = bitarray()
        self.log_file = open(log_file, 'w') if log_file else None
        self.indent = 0
        
        # We start by adding some metadata to the compressed file
        self.write(len(self.feature_map_shape), int2ba, 8, msg='Feature map shape, # dimensions: ')
        self.write(self.feature_map_shape, intarr2ba, 16, msg='Feature map shape: ')
        self.write(self.feature_map_max_val, float2ba, 32, msg='Feature map max val: ')
        self.write(self.quantization_bits, int2ba, 8, msg='Quantization bits: ')
        self.write(int(self.do_count_min), int2ba, 8, msg='Count Min sketch? ')
        self.write(int(self.do_bloomier), int2ba, 8, msg='Bloomier filter? ')
        if self.do_count_min:
            self.write(self.bloom_hash_count, int2ba, 8, msg='Bloom hash count: ')
            self.write(self.cm_width, int2ba, 16, msg='Count Min width: ')
            self.write(self.cm_depth, int2ba, 8, msg='Count Min depth: ')
        if self.do_bloomier:
            self.write(self.bloomier_hash_count, int2ba, 8, msg='Bloomier hash count: ')
            self.write(self.bloomier_second_table, int2ba, 8, msg='Bloomier second table? ')
        self.log_write()

        # Get indices and values of the nonzero entries in feature map
        indices, values = self.get_nonzero_vals(feature_map)
        # Quantize
        values = self.quantize(values)
        
        # Count Min sketch with Bloom filters
        if self.do_count_min:
            bloom, sketch = self.get_count_min(indices, values)
            sketch = sketch.flatten()
            bits_per_val = int(np.ceil(np.log2(sketch.max()+1)))
            if bloom is not None:
                self.write(len(bloom), int2ba, 16, msg='Bloom filter length: ')
                self.write(bloom, msg='Bloom filter:\n')
                self.log_write()
            self.write(bits_per_val, int2ba, 8, msg='Bits per entry in Count Min sketch: ')
            self.write(sketch, intarr2ba, bits_per_val, msg='Count Min sketch:\n')

        # Bloomier filter
        elif self.do_bloomier:
            max_val = self.bloomier_hash_count - 1 if self.bloomier_second_table else int(values.max())
            self.bloomier_num_slots = round(indices.shape[0] * self.bloomier_slots_per_key)
            self.bloomier_bits_per_slot = int(np.ceil(np.log2(max_val / self.bloomier_fpp)))
            table1, table2, hash_seed = self.get_bloomier(indices, values, max_val)
            self.write(self.bloomier_num_slots, int2ba, 16, msg='Bloomier number of slots: ')
            self.write(self.bloomier_bits_per_slot, int2ba, 8, msg='Bloomier bits per slot: ')
            self.write(max_val, int2ba, max(8, self.quantization_bits), msg='Bloomier max val: ')
            self.write(hash_seed, int2ba, 16, msg='Bloomier hash seed: ')
            self.write(table1, msg='Bloomier filter table 1:\n')
            if self.bloomier_second_table:
                self.write(table2, intarr2ba, self.quantization_bits, msg='\nBloomier filter table 2:\n')
        
        # Dictionary of nonzero elements
        else:
            num_elts = values.shape[0]
            deltas = np.ndarray(indices.shape, int)
            deltas[0] = indices[0]
            deltas[1:] = np.diff(indices)
            delta_num_bits = int(np.ceil(np.log2(deltas.max()+1)))
            self.write(num_elts, int2ba, 32, msg='Number of nonzero elements: ')
            self.write(delta_num_bits, int2ba, 8, msg='Bits to store each delta-encoded index: ')
            self.write(deltas, intarr2ba, delta_num_bits, msg='\nDelta-encoded indices of nonzero elments:\n')
            self.write(values, intarr2ba, self.quantization_bits, msg='\nValues of nonzero elments:\n')

        self.log_write('\n\n----------------------------------------------------------------\n')

        # Get what has already been written to compressed file, clear the compressed file,
        # and start fresh by writing whether Huffman encoding is to be used
        file = self.get_file()
        self.clear_file()
        self.write(int(self.do_huffman), int2ba, 8, msg='Huffman encoding? ')
        # Do Huffman coding on the file, if indicated
        if self.do_huffman:
            num_padding_bits, tree, encoded = self.huffman(file)
            tree_bitarray = tree2ba(tree)
            self.write(self.huffman_symbol_size, int2ba, 8, msg='Huffman symbol size: ')
            self.write(num_padding_bits, int2ba, 8, msg='Number of padding bits: ')
            self.write(len(tree_bitarray), int2ba, 16, msg='Number of bits in Huffman tree: ')
            self.write(len(encoded), int2ba, 32, msg='Number of bits in Huffman encoding: ')
            self.write(tree_bitarray, msg='\nHuffman tree as bits:\n')
            self.write(encoded, msg='\nHuffman encoding:\n')
        else:
            # Simply write file back
            self.write(file)

        # Write compressed file to disk
        with open(compressed_file, 'wb') as f:
            self.compressed_bitarray.tofile(f)
        
        if self.log_file:
            self.log_file.close()

    def quantize(self, flat_arr: np.ndarray) -> np.ndarray:
        target_min_val = 0
        target_max_val = 2**self.quantization_bits - 1
        scale = target_max_val / self.feature_map_max_val
        quantized_arr_float = flat_arr * scale
        quantized_arr_rounded = np.round(quantized_arr_float).astype(int)
        quantized_arr = np.clip(quantized_arr_rounded, target_min_val, target_max_val)
        return quantized_arr

    def get_nonzero_vals(self, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nonzero_mask = arr != 0
        nonzero_arr = arr[nonzero_mask]
        nonzero_idx = np.arange(arr.shape[0])[nonzero_mask]
        return nonzero_idx, nonzero_arr

    def get_count_min(self, indices: np.ndarray, values: np.ndarray) -> tuple[bitarray, np.ndarray]:
        # Bloom filter to store the nonzero indices
        if self.bloom_hash_count > 0:
            num_vals = values.shape[0]
            size = round(num_vals * self.bloom_bits_per_element)
            bloom = BloomFilter(size, self.bloom_hash_count)
            for idx in indices:
                bloom.add(idx)
            bloom_bit_array = bloom.bit_array
        else:
            # Bloom filter not used
            bloom_bit_array = None
        
        # Count Min sketch
        sketch = CountMinSketch(self.cm_width, self.cm_depth)
        for i, v in zip(indices, values):
            sketch.update(i, v)

        return bloom_bit_array, sketch.table

    def get_bloomier(self, indices: np.ndarray, values: np.ndarray, max_val: int) \
        -> tuple[bitarray, np.ndarray | None, int]:
        """
        Create a Bloomier filter that maps indices to their corresponding values.
        
        Args:
            indices: Array of indices (keys) from the feature map
            values: Array of non-zero values corresponding to the indices
            max_val: Maximum value stored in first table
            
        Returns:
            table1: First table of Bloomier filter
            table2: Second table of Bloomier filter (optional)
            hash_seed: Seed used for hash functions
        """
        # Initialize the Bloomier filter
        bloomier = BloomierFilter(m=self.bloomier_num_slots,
                                  k=self.bloomier_hash_count,
                                  q=self.bloomier_bits_per_slot,
                                  second_table=self.bloomier_second_table,
                                  max_val=max_val)
        # Create a dictionary mapping indices to values
        assignments = {int(idx): int(val) for idx, val in zip(indices, values)}
        # Build the filter
        bloomier.create(assignments)
        # Return the filter
        table1 = sum((bits for bits in bloomier.table1), bitarray())  # Concatenate all the bits
        table2 = np.array(bloomier.table2) if self.bloomier_second_table else None
        return table1, table2, bloomier.hash_seed

    def huffman(self, bits: bitarray):
        def build_codes(node: Node, prefix=bitarray(), codebook: dict[bitarray, bitarray] = {}):
            if node.symbol is not None:
                codebook[node.symbol] = prefix
            else:
                build_codes(node.left, prefix + bitarray('0'), codebook)
                build_codes(node.right, prefix + bitarray('1'), codebook)
            return codebook

        remainder = len(bits) % self.huffman_symbol_size
        if remainder == 0:
            num_padding_bits = 0
        else:
            num_padding_bits = self.huffman_symbol_size - remainder
        bits_padded = bits.to01() + '0'*num_padding_bits
        data = [bits_padded[i : i+self.huffman_symbol_size] for i in range(0, len(bits_padded), self.huffman_symbol_size)]
        frequency = Counter(data)
        heap = [Node(freq, sym, None, None) for sym, freq in frequency.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            node1 = heapq.heappop(heap)
            node2 = heapq.heappop(heap)
            merged = Node(node1.freq + node2.freq, None, node1, node2)
            heapq.heappush(heap, merged)
        tree = heap[0]
        codebook = build_codes(tree)
        encoded = bitarray()
        for symbol in data:
            encoded.extend(codebook[symbol])
        return num_padding_bits, tree, encoded
    
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

    def get_file(self):
        return self.compressed_bitarray.copy()
    
    def clear_file(self):
        self.compressed_bitarray.clear()


if __name__ == '__main__':
    compressor = FeatureMapCompressor(
        feature_map_options=FeatureMapOptions(shape=(1,4096)),
        quantization_options=QuantizationOptions(quantization_bits=4),
        # count_min_options=CountMinOptions(bloom_fpp=0.01, cm_epsilon=0.01, cm_delta=0.001),
        # bloomier_options=BloomierOptions(fpp=0.2, slots_per_key=1.3, hash_count=3, second_table=False),
        huffman_options=HuffmanOptions(symbol_size=4),
    )
    compressor.compress_feature_map('feature_vector_nparray_4096.npy', 'compressed', 'log_compressor.txt')
