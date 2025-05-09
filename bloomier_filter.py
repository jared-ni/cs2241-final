import numpy as np
import hashlib
import mmh3
from bitarray import bitarray
from collections import OrderedDict
from typing import List, Dict, Tuple, OrderedDict
from helper import *

class BloomierFilter:
    def __init__(
            self,
            m: int,
            k: int,
            q: int,
            second_table: bool,
            max_val: int,
            table1: List[bitarray] | None = None,
            table2: List[int] | None = None,
            hash_seed: int | None = None,
    ):
        """
        Immutable Bloomier Filter.

        Args:
            m (int): Number of table slots
            k (int): Number of hash functions
            q (int): Number of bits per slot (bitwidth)
            second_table (bool): Whether to use a second table (mutable Bloom filter)
            max_val (int): Maximum value stored in first table
            table1, table2 (optional np.ndarrays): Load from existing tables
            hash_seed (optional int): Provide seed for hash functions when loading from existing table
        """
        self.m = m
        self.k = k
        self.q = q
        self.table1 = [bitarray('0'*self.q) for _ in range(self.m)] if table1 is None else table1
        if second_table:
            self.table2 = [0 for _ in range(self.m)] if table2 is None else table2
        else:
            self.table2 = None
        self.max_val = max_val
        if hash_seed is not None:
            self.hash_seed = hash_seed

    def _hashes(self, key: int) -> Tuple[List[int], bitarray]:
        """
        Generate k table indices (a neighborhood) and a q-bit mask for a given key.
        """
        neighborhood = [mmh3.hash(str(key), self.hash_seed + i) % self.m for i in range(self.k)]

        key_bytes = (str(key) + str(self.hash_seed)).encode()
        digest = hashlib.sha256(key_bytes).digest()
        mask = bitarray()
        mask.frombytes(digest)
        mask = mask[:self.q]

        return neighborhood, mask

    def _encode(self, value: int) -> bitarray:
        return int2ba(value, self.q)
    
    def _decode(self, bits: bitarray) -> int | None:
        value = ba2int(bits)
        return value if value <= self.max_val else None

    def _tweak(self, key: int, neighborhoods: Dict[int, List[int]]) -> int | None:
        """
        Finds the index of the first table slot h in the neighborhood of `key`
        such that h is in no other neighborhood. Returns None if this doesn't exist.
        """
        # All values in the other neighborhoods
        others = set(x for k, n in neighborhoods.items() for x in n if k != key)
        for i, h in enumerate(neighborhoods[key]):
            if h not in others:
                return i
        return None

    def _find_match(self, neighborhoods: Dict[int, List[int]]) -> OrderedDict[int, int] | None:
        """
        Greedy algorithm to assign a unique index in each neighborhood.

        `neighborhoods` maps each key to its neighborhood, expressed as a list of table slot indices.
        The goal is to match each key to exactly one table slot of its neighborhood,
        such that no key is mapped to the same slot as any another key.

        Returns:
            - A matching of each key to the index of the chosen item in its neighborhood,
              with the keys given in a particular ordering.
        Returns None if algorithm fails to find a matching.
        """
        # Greedily assign non-conflicting table slots to the keys
        matching: OrderedDict[int, int] = OrderedDict()
        if not neighborhoods:
            return matching
        easy_matches = []
        for k in neighborhoods.keys():
            idx = self._tweak(k, neighborhoods)
            if idx is not None:
                matching[k] = idx
                easy_matches.append(k)
        if not easy_matches:
            return None
        
        # Recursive call for remaining unmatched keys
        neighborhoods = {k: v for k, v in neighborhoods.items() if k not in easy_matches}
        matching_new = self._find_match(neighborhoods)
        if matching_new is None:
            return None

        # Move the keys in the initial "easy matches" to the end of the ordering
        # (Inserting into OrderedDict adds to end)
        for k in easy_matches:
            matching_new[k] = matching[k]

        return matching_new

    def create(self, assignments: Dict[int, int]) -> None:
        """
        Build the immutable Bloomier filter.
        Args:
            assignments: Dictionary mapping keys to their values
        """
        self.hash_seed = 0
        attempts = 1
        while True:
            neighborhoods: Dict[int, List[int]] = {}
            masks: Dict[int, bitarray] = {}
            # For each key, use the hash functions to
            # find the list of table slot indices that the key maps to (the key's neighborhood)
            # and the q-bit mask representing the key
            for k in assignments.keys():
                neighborhoods[k], masks[k] = self._hashes(k)
            matching = self._find_match(neighborhoods)
            # Successfully found a matching from keys to table slots
            if matching is not None:
                break
            if attempts == 100:
                raise Exception('100 failed attempts at constructing Bloomier filter.')
            # Otherwise, try again with new hash functions
            self.hash_seed += self.k
            attempts += 1
            
        # Iterate in order through the matchings between keys and slots
        # and construct the table
        for k, neighborhood_idx in matching.items():
            # Neighborhood of the given key
            neighborhood = neighborhoods[k]
            # Convert the neighborhood index of the matching slot into a table slot index
            table_idx = neighborhood[neighborhood_idx]
            # XOR together the encoding of the value assigned to the key, the key's mask,
            # and all the other table slots in the neighborhood
            xor_sum = self._encode(neighborhood_idx if self.table2 else assignments[k])
            xor_sum ^= masks[k]
            for i, h in enumerate(neighborhood):
                if i != neighborhood_idx:
                    xor_sum ^= self.table1[h]
            self.table1[table_idx] = xor_sum
            if self.table2:
                self.table2[table_idx] = assignments[k]

    def query(self, key: int):
        neighborhood, mask = self._hashes(key)
        xor_sum = mask
        for i, h in enumerate(neighborhood):
            xor_sum ^= self.table1[h]
        if self.table2:
            neighborhood_idx = self._decode(xor_sum)
            if neighborhood_idx is None:
                return None
            table_idx = neighborhood[neighborhood_idx]
            return self.table2[table_idx]
        else:
            value = self._decode(xor_sum)
            return value if value is not None else None
