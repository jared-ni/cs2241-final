import numpy as np
import random
from typing import List, Tuple, Dict, Any
import hashlib
import networkx as nx
from bitarray import bitarray

class BloomierFilter:
    def __init__(self, m: int, k: int, q: int):
        """
        Immutable Bloomier Filter that exactly follows the construction from Chazelle et al. (2004).

        Args:
            m (int): Number of table entries
            k (int): Number of hash functions
            q (int): Number of bits per value (bitwidth)
        """
        self.m = m
        self.k = k
        self.q = q
        self.table = np.zeros((m, q), dtype=np.uint8)
        self.hash_seed = 7

    def _hashes(self, key: int) -> Tuple[List[int], np.ndarray]:
        """
        Generate k table indices and a q-bit mask for a given key.
        """
        key_bytes = (str(key) + str(self.hash_seed)).encode()
        digest = hashlib.sha256(key_bytes).digest()
        mask = bitarray()
        mask.frombytes(digest)
        mask = mask[:self.q].tolist()
        mask = np.array(mask)

        indices = []
        for i in range(self.k):
            h = int.from_bytes(digest[self.q + i*4:self.q + (i+1)*4], 'big') % self.m
            indices.append(h)

        return indices, mask

    def _encode(self, value: int) -> np.ndarray:
        """
        Encode an integer value into a q-bit vector.
        """
        return np.array([(value >> i) & 1 for i in range(self.q)], dtype=np.uint8)

    def _decode(self, bits: np.ndarray, max_value: int) -> Any:
        """
        Decode q-bit vector into integer, or return None if invalid.
        """
        val = 0
        for i in range(self.q):
            val |= (bits[i] << i)
        return val if val < max_value else None

    def _find_matching(self, keys: List[int], neighborhoods: Dict[int, List[int]]) -> Dict[int, int]:
        """
        Greedy algorithm to assign a unique index in each neighborhood.
        """
        # assigned = set()
        # matching = {}
        # for key in keys:
        #     for idx in neighborhoods[key]:
        #         if idx not in assigned:
        #             matching[key] = idx
        #             assigned.add(idx)
        #             break
        #     else:
        #         raise ValueError(f"Could not find non-conflicting match for key {key}")
        # return matching

        G = nx.Graph()
        for key in keys:
            for idx in neighborhoods[key]:
                G.add_edge(f"k_{key}", f"t_{idx}")

        matching = nx.bipartite.maximum_matching(G, top_nodes={f"k_{key}" for key in keys})
        assignment = {}
        for key in keys:
            match = matching.get(f"k_{key}")
            if match is None:
                raise ValueError(f"Could not find non-conflicting match for key {key}")
            assigned_idx = int(match.split("_", 1)[1])
            assignment[key] = assigned_idx

        return assignment

    def build(self, assignments: Dict[int, int]) -> None:
        """
        Build the immutable Bloomier filter.
        Args:
            assignments: Dictionary mapping keys to their values
        """
        keys = list(assignments.keys())
        neighborhoods = {}

        for key in keys:
            h, _ = self._hashes(key)
            neighborhoods[key] = h

        matching = self._find_matching(keys, neighborhoods)

        for key in keys:
            value = assignments[key]
            encoded_val = self._encode(value)
            h_indices, mask = self._hashes(key)
            assigned = matching[key]

            xor_sum = mask.copy()
            for idx in h_indices:
                xor_sum ^= self.table[idx]
            xor_sum ^= self.table[assigned]  # Cancel the double-counted assigned idx
            self.table[assigned] = xor_sum ^ encoded_val

    def query(self, key: int, max_value: int) -> Any:
        """
        Query the Bloomier filter.
        Args:
            key: Key to look up
            max_value: Largest allowed value in the domain of f
        Returns:
            The associated value or None
        """
        h_indices, mask = self._hashes(key)
        xor_sum = mask.copy()
        for idx in h_indices:
            xor_sum ^= self.table[idx]
        return self._decode(xor_sum, max_value)
