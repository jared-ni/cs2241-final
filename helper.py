from bitarray import bitarray
from bitarray.util import int2ba, ba2int
import struct
import numpy as np

def float2ba(value: float) -> bitarray:
    """Convert the given 32-bit float to a bitarray"""
    bits = bitarray()
    packed = struct.pack('>f', value)
    bits.frombytes(packed)
    return bits

def ba2float(bits: bitarray) -> float:
    """Convert the given bitarray to a 32-bit float"""
    return struct.unpack('>f', bits.tobytes())[0]

def intarr2ba(arr: np.ndarray, length: int, signed: bool = False) -> bitarray:
    """
    Convert the given integer array to a bitarray.
    Each integer is represented using `length` bits and may be signed or unsigned.
    """
    bits = bitarray()
    for x in arr:
        bits += int2ba(int(x), length, signed=signed)
    return bits

def ba2intarr(bits: bitarray, length: int, signed: bool = False) -> np.ndarray:
    """
    Convert the given bitarray to an integer array.
    Each integer is represented using `length` bits and may be signed or unsigned.
    """
    arr = []
    for i in range(len(bits) // length):
        arr.append(ba2int(bits[i:i+length], signed=signed))
    return np.array(arr)
