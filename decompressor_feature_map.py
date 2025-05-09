import numpy as np
from bitarray import bitarray
from helper import *
from bloom import BloomFilter
from bloomier_filter import BloomierFilter
from count_min_sketch import CountMinSketch
from compressor_base_feature_map import FeatureMapCompressorBase
from typing import Any, Callable
from math import prod


class FeatureMapDecompressor(FeatureMapCompressorBase):
    def __init__(self):
        super().__init__()

    def decompress_image(self, compressed_file: str, output_file: str, log_file: str | None = None):
        self.pos = 0
        self.indent = 0

        self.compressed_bitarray = bitarray()
        with open(compressed_file, 'rb') as f:
            self.compressed_bitarray.fromfile(f)

        self.log_file = open(log_file, 'w') if log_file else None

        # Undo Huffman encoding, if used
        self.do_huffman = bool(self.read(8, ba2int, msg='Huffman encoding? '))
        if self.do_huffman:
            self.huffman_symbol_size = self.read(8, ba2int, msg='Huffman symbol size: ')
            num_padding_bits = self.read(8, ba2int, msg='Number of padding bits: ')
            tree_bitarray_len = self.read(16, ba2int, msg='Number of bits in Huffman tree: ')
            encoded_len = self.read(32, ba2int, msg='Number of bits in Huffman encoding: ')
            tree_bitarray = self.read(tree_bitarray_len, msg='\nHuffman tree as bits:\n')
            encoded = self.read(encoded_len, msg='\nHuffman encoding:\n')
            # Decode Huffman encoding and set the file to be this decoded version for the following steps
            self.set_file(self.huffman_undo(num_padding_bits, tree_bitarray, encoded))
        
        self.log_write('\n\n----------------------------------------------------------------\n')

        # Read metadata at start of file
        feature_map_shape_len = self.read(8, ba2int, msg='Feature map shape, # dimensions: ')
        self.feature_map_shape = self.read(16*feature_map_shape_len, ba2tuple, 16, msg='Feature map shape: ')
        self.feature_map_max_val = self.read(32, ba2float, msg='Feature map max val: ')
        self.quantization_bits = self.read(8, ba2int, msg='Quantization bits: ')
        self.do_count_min = bool(self.read(8, ba2int, msg='Count Min sketch? '))
        self.do_bloomier = bool(self.read(8, ba2int, msg='Bloomier filter? '))
        if self.do_count_min:
            self.bloom_hash_count = self.read(8, ba2int, msg='Bloom hash count: ')
            self.cm_width = self.read(16, ba2int, msg='Count Min width: ')
            self.cm_depth = self.read(8, ba2int, msg='Count Min depth: ')
        if self.do_bloomier:
            self.bloomier_hash_count = self.read(8, ba2int, msg='Bloomier hash count: ')
            self.bloomier_second_table = self.read(8, ba2int, msg='Bloomier second table? ')
        self.log_write()

        # Total number of entries in feature map across all dimensions
        self.feature_map_size = prod(self.feature_map_shape)

        # Count Min sketch with Bloom filters
        if self.do_count_min:
            if self.bloom_hash_count > 0:
                bloom_len = self.read(16, ba2int, msg='Bloom filter length: ')
                bloom = self.read(bloom_len, msg='Bloom filter:\n')
                self.log_write()
            else:
                bloom = None
            bits_per_val = self.read(8, ba2int, msg='Bits per entry in Count Min sketch: ')
            sketch_bits = self.cm_width * self.cm_depth * bits_per_val
            sketch = self.read(sketch_bits, ba2intarr, bits_per_val, msg='Count Min sketch:\n')
            sketch = sketch.reshape(self.cm_depth, self.cm_width)
            arr = self.count_min_undo(bloom, sketch)

        # Bloomier filter
        elif self.do_bloomier:
            self.bloomier_num_slots = self.read(16, ba2int, msg='Bloomier number of slots: ')
            self.bloomier_bits_per_slot = self.read(8, ba2int, msg='Bloomier bits per slot: ')
            max_val = self.read(max(8, self.quantization_bits), ba2int, msg='Bloomier max val: ')
            hash_seed = self.read(16, ba2int, msg='Bloomier hash seed: ')
            num_bits1 = self.bloomier_num_slots * self.bloomier_bits_per_slot
            num_bits2 = self.bloomier_num_slots * self.quantization_bits
            table1 = self.read(num_bits1, msg='Bloomier filter table 1:\n')
            if self.bloomier_second_table:
                table2 = self.read(num_bits2, ba2intarr, self.quantization_bits,
                                   msg='\nBloomier filter table 2:\n')
            else:
                table2 = None
            arr = self.bloomier_undo(table1, table2, max_val, hash_seed)
        
        # Dictionary of nonzero elements
        else:
            num_elts = self.read(32, ba2int, msg='Number of nonzero elements: ')
            delta_num_bits = self.read(8, ba2int, msg='Bits to store each delta-encoded index: ')
            deltas = self.read(num_elts*delta_num_bits, ba2intarr, delta_num_bits,
                               msg='\nDelta-encoded indices of nonzero elments:\n')
            values = self.read(num_elts*self.quantization_bits, ba2intarr, self.quantization_bits,
                               msg='\nValues of nonzero elments:\n')
            indices = deltas.cumsum()
            arr = np.zeros(self.feature_map_size)
            arr[indices] = values

        # Undo quantization step and unflatten
        arr = self.quantize_undo(arr)
        arr = arr.reshape(self.feature_map_shape)

        # Save image to file
        np.save(output_file, arr)

        if self.log_file:
            self.log_file.close()

    def quantize_undo(self, arr: np.ndarray) -> np.ndarray:
        target_max_val = 2**self.quantization_bits - 1
        return arr * self.feature_map_max_val / target_max_val
    
    def count_min_undo(self, bloom_table: np.ndarray, sketch_table: np.ndarray) -> np.ndarray:
        if bloom_table is not None:
            bloom = BloomFilter(bloom_table, self.bloom_hash_count)
        sketch = CountMinSketch(self.cm_width, self.cm_depth, sketch_table)
        arr = np.zeros(self.feature_map_size, dtype=int)
        for i in range(self.feature_map_size):
            if bloom_table is None or bloom.check(i):
                arr[i] = sketch.query(i)
        return arr

    def bloomier_undo(
            self,
            table1: bitarray,
            table2: np.ndarray | None,
            max_val: int,
            hash_seed: int
    ) -> np.ndarray:
        t1 = []
        for i in range(0, len(table1), self.bloomier_bits_per_slot):
            t1.append(table1[i : i + self.bloomier_bits_per_slot])
        t2 = table2.tolist() if self.bloomier_second_table else None
        bloomier = BloomierFilter(m=self.bloomier_num_slots,
                                  k=self.bloomier_hash_count,
                                  q=self.bloomier_bits_per_slot,
                                  second_table=self.bloomier_second_table,
                                  table1=t1,
                                  table2=t2,
                                  max_val=max_val,
                                  hash_seed=hash_seed)
        arr = np.zeros(self.feature_map_size, dtype=int)
        for i in range(self.feature_map_size):
            val = bloomier.query(i)
            if val is not None:
                arr[i] = val
        return arr

    def huffman_undo(self, num_padding_bits: int, tree_bitarray: bitarray, encoded: bitarray) -> bitarray:
        tree = ba2tree(tree_bitarray, self.huffman_symbol_size)
        decoded = bitarray()
        node = tree
        for bit in encoded:
            node = node.left if bit == 0 else node.right
            if node.symbol is not None:
                decoded += bitarray(node.symbol)
                node = tree
        return decoded[:-num_padding_bits] if num_padding_bits > 0 else decoded

    def read(self,
             num_bits: int,
             func: Callable | None = None,
             length: int | None = None,
             signed: bool = False,
             msg: Any = None):
        """
        Read the next `num_bits` bits from `self.compressed_bitarray`,
        converted into a value using `func` (or raw bitarray if `func` is None).
        Pass `length` and `signed` arguments to `func` if necessary.
        If `msg` is provided and there is a log file, write `msg` followed by `val` to the log file.
        """
        bits = self.read_file(num_bits)
        val = func(bits, length, signed) if func else bits
        if msg is not None:
            str_val = self.log_as_str(val)
            self.log_write(str(msg) + str_val)
        return val

    
    def read_file(self, length: int) -> bitarray:
        """
        Read the next `length` bits from `self.compressed_bitarray`.
        """
        bits = self.compressed_bitarray[self.pos : self.pos+length]
        self.pos += length
        return bits
    
    def set_file(self, bits: bitarray):
        self.compressed_bitarray = bits
        self.pos = 0

