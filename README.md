# cs2241-final

This repository contains the code for our CS 2241 final project, "Learning from Less: Feature-Level Image Compression with Probabilistic Data Structures".

The main compression and decompression algorithms are implemented in `compressor_feature_map.py` and `decompressor_feature_map.py`.

We used `run_experiments.py` and `run_outline_experiments.py` to call the compression algorithms and run them on feature maps and Sobel operator outlines, respectively.

The Jupyter notebooks contain the model training and inference portions of our experiments.

The Bloom filter, Bloomier filter, and Count-Min sketch are implemented in `bloom.py`, `bloomier_filter.py`, and `count_min_sketch.py`.

The Sobel operator for edge detection is implemented in `outline.py`.

`helper.py` contains some utilities for packing data into bit strings as well as for Huffman coding.

`compressor.py` and `decompressor.py` implement the algorithms described in Appendix A.
