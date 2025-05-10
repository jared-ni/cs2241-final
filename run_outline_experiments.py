from compressor_feature_map import FeatureMapCompressor, FeatureMapOptions, QuantizationOptions, CountMinOptions, BloomierOptions, HuffmanOptions
from decompressor_feature_map import FeatureMapDecompressor
import numpy as np
import os

if __name__ == "__main__":
    # generate compressed feature maps for the count-min sketch
    compressor = FeatureMapCompressor(
        feature_map_options=FeatureMapOptions(shape=(224,224)),
        quantization_options=QuantizationOptions(quantization_bits=2),
        count_min_options=CountMinOptions(bloom_fpp=0.1, cm_epsilon=0.1, cm_delta=0.1),
        # bloomier_options=BloomierOptions(fpp=0.3, slots_per_key=1.3, hash_count=3, second_table=False),
        huffman_options=HuffmanOptions(symbol_size=4),
    )
    decompressor = FeatureMapDecompressor()

    current_directory = os.path.dirname(os.path.abspath(__file__))
    base_directory = os.path.join(current_directory, 'experiments', 'kaggle-animals10', 
                                  'kaggle-animals10_10k_outlines_npy') 
    total_files_processed = 0

    output_directory = os.path.join(current_directory, 'experiments', 'kaggle-animals10', 
                                    '4-outline-experiments', 'animals10_10k_outline-quant2bit_count-min_fpp01_0101')
    
    animals = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 
               'elephant', 'horse', 'sheep', 'spider', 'squirrel']
    # create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if not os.path.exists(f"{output_directory}/compressed"):
        os.makedirs(f"{output_directory}/compressed")
        os.makedirs(f"{output_directory}/compressed/raw-img")
        for animal in animals:
            os.makedirs(f"{output_directory}/compressed/raw-img/{animal}")

    if not os.path.exists(f"{output_directory}/decompressed"):
        os.makedirs(f"{output_directory}/decompressed")
        os.makedirs(f"{output_directory}/decompressed/raw-img")
        for animal in animals:
            os.makedirs(f"{output_directory}/decompressed/raw-img/{animal}")
    
    for split in ['raw-img']:
        for class_name in ['butterfly', 'cat', 'chicken', 'cow', 'dog', 
                           'elephant', 'horse', 'sheep', 'spider', 'squirrel']:
            # delete 60% of the files in each class
            files = os.listdir(os.path.join(base_directory, split, class_name))
            print(f"Number of files in {class_name}: {len(files)}")
            for file in files:
                cur_feature_map = os.path.join(base_directory, split, class_name, file)
                total_files_processed += 1
                # print name of the cur_feature_map
                print(f"Current feature map: {cur_feature_map}")
                # load the feature map
                compressor.compress_feature_map(cur_feature_map, f'{output_directory}/compressed/{split}/{class_name}/{file}')
                                                # f'{output_directory}/log_compressor_{file}.txt')
                
                decompressor.decompress_image(f'{output_directory}/compressed/{split}/{class_name}/{file}', f'{output_directory}/decompressed/{split}/{class_name}/{file}')
                                                # f'{output_directory}/log_decompressor_{file}.txt')
                print("total files processed: ", total_files_processed)
    
            
