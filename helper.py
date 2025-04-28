from bitarray import bitarray
import bitarray.util as bitarray_util
import struct
import numpy as np
from typing import Literal

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

def intarr2ba(arr: np.ndarray, length: int, signed: bool = False) -> bitarray:
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
