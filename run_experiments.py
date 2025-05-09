from compressor_feature_map import FeatureMapCompressor, FeatureMapOptions, QuantizationOptions, CountMinOptions, BloomierOptions, HuffmanOptions
from decompressor_feature_map import FeatureMapDecompressor
import numpy as np
import os

if __name__ == "__main__":
    # generate compressed feature maps for the count-min sketch
    compressor = FeatureMapCompressor(
        feature_map_options=FeatureMapOptions(shape=(4096,)),
        quantization_options=QuantizationOptions(quantization_bits=8),
        count_min_options=CountMinOptions(bloom_fpp=0.01, cm_epsilon=0.01, cm_delta=0.001),
        # bloomier_options=BloomierOptions(fpp=0.01, slots_per_key=1.3, hash_count=3),
        huffman_options=HuffmanOptions(symbol_size=4),
    )
    decompressor = FeatureMapDecompressor()

    current_directory = os.path.dirname(os.path.abspath(__file__))
    base_directory = os.path.join(current_directory, 'experiments', 'kaggle', 'feature_vectors_5k') 
    total_files_processed = 0

    output_directory = os.path.join(current_directory, 'experiments', '_5k')

    # create the output directory if it doesn't exist
    if not os.path.exists(f"{output_directory}/compressed"):
        os.makedirs(f"{output_directory}/compressed")

        os.makedirs(f"{output_directory}/compressed/test")
        os.makedirs(f"{output_directory}/compressed/test/Dog")
        os.makedirs(f"{output_directory}/compressed/test/Cat")

        os.makedirs(f"{output_directory}/compressed/train")
        os.makedirs(f"{output_directory}/compressed/train/Dog")
        os.makedirs(f"{output_directory}/compressed/train/Cat")

        os.makedirs(f"{output_directory}/compressed/val")
        os.makedirs(f"{output_directory}/compressed/val/Dog")
        os.makedirs(f"{output_directory}/compressed/val/Cat")
    if not os.path.exists(f"{output_directory}/decompressed"):
        os.makedirs(f"{output_directory}/decompressed")

        os.makedirs(f"{output_directory}/decompressed/test")
        os.makedirs(f"{output_directory}/decompressed/test/Dog")
        os.makedirs(f"{output_directory}/decompressed/test/Cat")

        os.makedirs(f"{output_directory}/decompressed/train")
        os.makedirs(f"{output_directory}/decompressed/train/Dog")
        os.makedirs(f"{output_directory}/decompressed/train/Cat")

        os.makedirs(f"{output_directory}/decompressed/val")
        os.makedirs(f"{output_directory}/decompressed/val/Dog")
        os.makedirs(f"{output_directory}/decompressed/val/Cat")
    

    for split in ['test', 'train', 'val']:
        for class_name in ['Cat', 'Dog']:
            # iterate through the files in the directory
            # for file in os.listdir(base_directory + '/' + split + '/' + class_name):
            for file in os.listdir(os.path.join(base_directory, split, class_name)):
                cur_feature_map = os.path.join(base_directory, split, class_name, file)
                total_files_processed += 1
                # print name of the cur_feature_map
                print(f"Current feature map: {cur_feature_map}")
                # load the feature map
                # cur_feature_map = np.load(cur_feature_map)
                # # print the shape of the feature map
                # print(f"Feature map shape: {cur_feature_map.shape}")
                compressor.compress_feature_map(cur_feature_map, f'{output_directory}/compressed/{split}/{class_name}/{file}')
                                                # f'{output_directory}/log_compressor_{file}.txt')
                
                decompressor.decompress_image(f'{output_directory}/compressed/{split}/{class_name}/{file}', f'{output_directory}/decompressed/{split}/{class_name}/{file}')
                                                # f'{output_directory}/log_decompressor_{file}.txt')
                print("total files processed: ", total_files_processed)
                # with open(f'{output_directory}/decompressed.txt', 'w') as f:
                #     np.set_printoptions(threshold=np.inf)
                #     f.write(str(np.load('decompressed.npy').tolist()))
                
                # exit(0)

    # # go into base_directory/kaggle
    # compressor.compress_feature_map('feature_vector_nparray_4096.npy', 'compressed', 'log_compressor.txt')
    # decompressor.decompress_image('compressed', 'decompressed.npy', 'log_decompressor.txt')
    # with open('decompressed.txt', 'w') as f:
    #     np.set_printoptions(threshold=np.inf)
    #     f.write(str(np.load('decompressed.npy').tolist()))

    
