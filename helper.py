from bitarray import bitarray
import bitarray.util as bitarray_util
import struct
import numpy as np
from typing import Literal, Iterable

def int2ba(value: int, length: int, signed: bool = False):
    """Convert the given int to a bitarray"""
    return bitarray_util.int2ba(value, length, signed=signed)

def ba2int(bits: bitarray, length: Literal[None] = None, signed: bool = False):
    """Convert the given bitarray to an int"""
    return bitarray_util.ba2int(bits, signed)

def float2ba(value: float, length: Literal[32] = 32, signed: Literal[True] = True) -> bitarray:
    """Convert the given 32-bit float to a bitarray"""
    bits = bitarray()
    packed = struct.pack('>f', value)
    bits.frombytes(packed)
    return bits

def ba2float(bits: bitarray, length: Literal[32] = 32, signed: Literal[True] = True) -> float:
    """Convert the given bitarray to a 32-bit float"""
    return struct.unpack('>f', bits.tobytes())[0]

def intarr2ba(arr: Iterable[int], length: int, signed: bool = False) -> bitarray:
    """
    Convert the given integer array to a bitarray.
    Each integer is represented using `length` bits and may be signed or unsigned.
    """
    bits = bitarray()
    for x in arr:
        bits += int2ba(int(x), length, signed)
    return bits

def ba2intarr(bits: bitarray, length: int, signed: bool = False) -> np.ndarray:
    """
    Convert the given bitarray to an integer array.
    Each integer is represented using `length` bits and may be signed or unsigned.
    """
    arr = []
    for i in range(0, len(bits), length):
        arr.append(ba2int(bits[i:i+length], None, signed))
    return np.array(arr)

def ba2tuple(bits: bitarray, length: int, signed: bool = False) -> np.ndarray:
    """
    Convert the given bitarray to an integer tuple.
    Each integer is represented using `length` bits and may be signed or unsigned.
    """
    return tuple(ba2intarr(bits, length, signed))

class Node:
    def __init__(self, freq=None, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
    def __lt__(self, other):
        return self.freq < other.freq
    
def tree2ba(node: Node, length: None = None, signed: None = None) -> bitarray:
    """
    Convert the given Huffman tree to a bitarray.
    Each node's symbol is represented using `length` bits.
    """
    def _tree2ba(node: Node, bits: bitarray):
        if node.symbol is not None:
            # Leaf node -> write symbol
            bits += bitarray('1') + bitarray(node.symbol)
        else:
            # Internal node -> recursively write child nodes
            bits += bitarray('0')
            _tree2ba(node.left, bits)
            _tree2ba(node.right, bits)
    bits = bitarray()
    _tree2ba(node, bits)
    return bits

def ba2tree(bits: bitarray, length: int, signed: None = None) -> Node:
    """
    Convert the given bitarray to a Huffman tree.
    Each node's symbol is represented using `length` bits.
    """
    def _ba2tree(pos: int):
        is_leaf = bits[pos]
        pos += 1
        if is_leaf:
            symbol = bits[pos : pos+length].to01()
            pos += length
            return Node(None, symbol, None, None), pos
        else:
            left, pos = _ba2tree(pos)
            right, pos = _ba2tree(pos)
            return Node(None, None, left, right), pos
    node, _ = _ba2tree(0)
    return node
