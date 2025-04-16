from bitarray import bitarray
from bitarray.util import int2ba, ba2int
import struct
import numpy as np

def float2ba(value: float) -> bitarray:
    bits = bitarray()
    packed = struct.pack('>f', value)
    bits.frombytes(packed)
    return bits

def ba2float(bits: bitarray) -> float:
    return struct.unpack('>f', bits.tobytes())[0]

def intarr2ba(arr: np.ndarray, length: int, signed: bool = False) -> bitarray:
    bits = bitarray()
    for x in arr:
        bits += int2ba(x, length, signed=signed)
    return bits

def ba2intarr(bits: bitarray, length: int, signed: bool = False) -> np.ndarray:
    arr = []
    for i in range(len(bits) // length):
        arr.append(ba2int(bits[i:i+length], signed=signed))
    return np.array(arr)
