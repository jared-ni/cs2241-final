import numpy as np
from typing import List, Tuple, Dict, Any
import hashlib
import networkx as nx
# from bitarray import bitarray # numpy is fine for fixed-width bit vectors

class BloomierFilter:
    def __init__(self, m: int, k: int, q_val: int, q_l: int | None = None, fixed_hash_seeds: List[int] | None = None): # Added fixed_hash_seeds
        self.m = m
        self.k = k
        self.q_val = q_val
        
        if q_l is None:
            self.q_l = (k - 1).bit_length() if k > 1 else 1 # Ensures q_l is at least 1
        else:
            self.q_l = q_l
        
        # Ensure q_l is at least 1, especially if k=1 ( (1-1).bit_length() = 0 )
        if self.q_l == 0:
            self.q_l = 1

        self.table1 = np.zeros((m, self.q_l), dtype=np.uint8)
        self.table2 = np.zeros((m, self.q_val), dtype=np.uint8)
        
        if fixed_hash_seeds is None:
            # For testing and reproducibility, use a fixed list of seeds
            self.hash_seeds = list(range(k + 1)) # Example: [0, 1, 2, ..., k]
            # Or use another specific list: e.g., [123, 456, 789, ...] up to k+1 elements
        else:
            self.hash_seeds = fixed_hash_seeds
        
        # Make sure you have enough seeds if k is large
        if len(self.hash_seeds) < k + 1:
             raise ValueError(f"Need at least {k+1} hash seeds, got {len(self.hash_seeds)}")
    
    def _get_hash_values(self, key_str: str) -> Tuple[List[int], np.ndarray]:
        indices = []
        mask_bits = np.zeros(self.q_l, dtype=np.uint8)

        # Generate mask from the (k)th seed (or 0th if you prefer to use one seed for multiple purposes)
        # Using self.hash_seeds[k] specifically for the mask ensures it's based on a different seed
        # than those strictly used for table indices, reducing correlations if k is small.
        mask_key_salted = (key_str + str(self.hash_seeds[self.k])).encode('utf-8') # Use the (k+1)th seed for mask
        mask_digest = hashlib.sha256(mask_key_salted).digest()
        
        for bit_pos in range(self.q_l):
            byte_idx = bit_pos // 8
            bit_in_byte_idx = bit_pos % 8
            if byte_idx < len(mask_digest):
                mask_bits[bit_pos] = (mask_digest[byte_idx] >> bit_in_byte_idx) & 1
        
        # Generate k hash values for indices
        for i in range(self.k):
            salted_key = (key_str + str(self.hash_seeds[i])).encode('utf-8') # Seeds 0 to k-1 for indices
            h_digest = hashlib.sha256(salted_key).digest()
            indices.append(int.from_bytes(h_digest[:4], 'big') % self.m)
            
        return indices, mask_bits


    def _encode_to_bits(self, value: int, num_bits: int) -> np.ndarray:
        if value < 0:
            raise ValueError("Value must be non-negative for simple bit encoding")
        if value >= (1 << num_bits):
            # Or handle more gracefully, e.g. by capping or erroring earlier
            print(f"Warning: Value {value} too large for {num_bits} bits. Will be truncated/incorrect.")
        
        bits = np.zeros(num_bits, dtype=np.uint8)
        for i in range(num_bits):
            if (value >> i) & 1:
                bits[i] = 1
        return bits

    def _decode_from_bits(self, bits: np.ndarray) -> int:
        val = 0
        for i in range(len(bits)):
            if bits[i]:
                val |= (1 << i)
        return val

    # Wrappers for clarity
    def _encode_l_idx(self, l_idx: int) -> np.ndarray:
        return self._encode_to_bits(l_idx, self.q_l)

    def _decode_l_idx(self, bits: np.ndarray) -> Any: # Returns int or None
        val = self._decode_from_bits(bits)
        return val if val < self.k else None # l_idx must be < k

    def _encode_value(self, value: int) -> np.ndarray:
        return self._encode_to_bits(value, self.q_val)

    def _decode_value(self, bits: np.ndarray, max_value: int) -> Any: # Returns int or None
        val = self._decode_from_bits(bits)
        # The max_value check for actual data values is often application-specific
        # or might not be needed if any q_val-bit pattern is a valid value.
        # For now, let's assume it's a simple range check.
        return val if val < max_value else None


    def _find_matching(self, keys_str: List[str], neighborhoods: Dict[str, List[int]]) -> Dict[str, int]:
        G = nx.Graph()
        # keys_str is now sorted if called from build after my change.

        # Deterministic node addition order for key_nodes
        for key_s in keys_str: # keys_str is sorted
            G.add_node(f"k_{key_s}", bipartite=0)
            # For table_idx nodes, ensure their addition is also somewhat ordered if it matters.
            # Sorting neighborhoods[key_s] ensures deterministic edge addition for a given key_s
            sorted_table_indices_for_key = sorted(neighborhoods[key_s])
            for table_idx in sorted_table_indices_for_key:
                G.add_node(f"t_{table_idx}", bipartite=1) # add_node handles duplicates gracefully
                G.add_edge(f"k_{key_s}", f"t_{table_idx}")
        
        key_side_nodes_for_matching = [f"k_{s}" for s in keys_str]

        try:
            # Given a deterministically built graph, maximum_matching should be deterministic
            # unless it has an internal tie-breaking that is random (unlikely for standard algos).
            matching_dict_from_nx = nx.bipartite.maximum_matching(G, top_nodes=key_side_nodes_for_matching)
        except nx.AmbiguousSolution:
             print("Ambiguous solution in matching from NetworkX, using greedy fallback.")
             assigned_indices_greedy = set()
             matching_dict_from_nx = {} # This will be our greedy result
             # Greedy matching also needs to be deterministic: iterate keys_str (which is sorted)
             for key_s_node in key_side_nodes_for_matching: # Iterates based on sorted keys_str
                 # key_s_val = key_s_node[2:] # Not needed here
                 found_match = False
                 # Iterate neighbors in a defined order (e.g., sorted numerically)
                 # G.neighbors gives an iterator; convert to list and sort for determinism
                 sorted_neighbors = sorted([int(n[2:]) for n in G.neighbors(key_s_node)])
                 for table_idx_val in sorted_neighbors:
                     if table_idx_val not in assigned_indices_greedy:
                         matching_dict_from_nx[key_s_node] = f"t_{table_idx_val}"
                         assigned_indices_greedy.add(table_idx_val)
                         found_match = True
                         break
                 if not found_match:
                     key_s_for_error = key_s_node[2:]
                     raise ValueError(f"Greedy fallback could not find non-conflicting match for key {key_s_for_error}")

        # Parsing the matching_dict_from_nx
        parsed_assignment = {}
        for key_s_build_loop in keys_str: # Iterate in sorted order
            key_node_graph = f"k_{key_s_build_loop}"
            
            table_node_graph = matching_dict_from_nx.get(key_node_graph)
            if table_node_graph is None or not table_node_graph.startswith("t_"):
                raise ValueError(f"Key {key_s_build_loop} (node {key_node_graph}) not found in matching or matched incorrectly. Got: {table_node_graph}. Full Matching: {matching_dict_from_nx}")
            
            assigned_idx_val = int(table_node_graph[2:]) # remove "t_"
            parsed_assignment[key_s_build_loop] = assigned_idx_val
        
        if len(parsed_assignment) != len(keys_str):
            missing_keys = set(keys_str) - set(parsed_assignment.keys())
            raise ValueError(f"Matching did not cover all keys. Expected {len(keys_str)}, got {len(parsed_assignment)}. Missing: {missing_keys}")
        return parsed_assignment

    def build(self, assignments: Dict[Any, int]) -> None:
        str_assignments = {str(k): v for k, v in assignments.items()}
        # Sort keys_as_str to ensure deterministic order for graph construction
        # and iteration, which can influence matching if multiple perfect matchings exist.
        keys_as_str = sorted(list(str_assignments.keys())) # <--- ADD SORTED() HERE
        
        neighborhoods = {}
        key_hashes_masks = {}

        for key_s in keys_as_str: # Iteration order is now deterministic
            h_indices, mask = self._get_hash_values(key_s)
            neighborhoods[key_s] = h_indices # h_indices order from _get_hash_values is deterministic
            key_hashes_masks[key_s] = (h_indices, mask)

        try:
            # The graph G in _find_matching will now be built deterministically
            # if keys_str (passed to it) is sorted.
            matching = self._find_matching(keys_as_str, neighborhoods)
        except ValueError as e:
            print(f"Failed to find matching: {e}")
            print("Consider increasing m (table size) or adjusting k (number of hashes).")
            raise

        # The loop for setting table1 and table2 entries will also be deterministic
        for key_s in keys_as_str:
            actual_value_to_store = str_assignments[key_s]
            h_indices, mask = key_hashes_masks[key_s]
            L_assigned = matching[key_s]

            try:
                l_idx = h_indices.index(L_assigned)
            except ValueError:
                raise RuntimeError(f"Internal error: Assigned location {L_assigned} not in hash locations {h_indices} for key {key_s}")

            encoded_l = self._encode_l_idx(l_idx)

            # Logic to set self.table1[L_assigned]
            # interim_xor_sum = mask.copy()
            # for h_val in h_indices:
            #     interim_xor_sum ^= self.table1[h_val]
            # self.table1[L_assigned] = self.table1[L_assigned] ^ interim_xor_sum ^ encoded_l
            
            # Simplified setting of table1[L] (more direct from paper)
            # T1[L] = M XOR (SUM_{h in N(key), h!=L} T1[h]) XOR encoded_l
            temp_xor_sum_for_l = mask.copy() # Start with M
            for h_val_idx in range(len(h_indices)):
                if h_val_idx != l_idx : # i.e., h_indices[h_val_idx] != L_assigned
                    temp_xor_sum_for_l ^= self.table1[h_indices[h_val_idx]]
            
            self.table1[L_assigned] = temp_xor_sum_for_l ^ encoded_l


            self.table2[L_assigned] = self._encode_value(actual_value_to_store)

    def query(self, key: Any, max_value_in_domain: int) -> Any:
        key_s = str(key)
        h_indices, mask = self._get_hash_values(key_s)
        
        retrieved_l_bits = mask.copy()
        for h_idx_val in h_indices:
            if h_idx_val >= self.m : # Should not happen with % m in hash
                print(f"Warning: hash index {h_idx_val} out of bounds for table size {self.m}")
                return None 
            retrieved_l_bits ^= self.table1[h_idx_val]
        
        l_decoded = self._decode_l_idx(retrieved_l_bits)

        if l_decoded is not None: # Check if l_decoded is a valid index for h_indices
            if l_decoded < len(h_indices): # l_decoded must be < k (which is len(h_indices))
                L_final = h_indices[l_decoded]
                if L_final < self.m: # Ensure L_final is within table2 bounds
                     return self._decode_value(self.table2[L_final], max_value=max_value_in_domain)
                else: # Should not happen
                    print(f"Warning: L_final index {L_final} out of bounds for table size {self.m}")
        
        return None

if __name__ == "__main__":
    # Example usage
    bloomier = BloomierFilter(m=20, k=4, q_val=8)
    assignments = {0: 30, 1: 41, 2: 59, 3: 123}
    # assignments = {0: 42, 1: 99, 2: 123, 3: 253}
    bloomier.build(assignments)

    for key in assignments.keys():
        value = bloomier.query(key, max_value_in_domain=255)
        print(f"Key: {key}, Value: {value}")
    
    for key in range(5, 30, 1):
        value = bloomier.query(key, max_value_in_domain=255)
        print(f"Key: {key}, Value: {value}")