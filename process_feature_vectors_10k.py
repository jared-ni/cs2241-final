import numpy as np
import os

if __name__ == "__main__":

    current_directory = os.path.dirname(os.path.abspath(__file__))
    # base_directory = os.path.join(current_directory, 'experiments', 'kaggle', 'feature_vectors_5k')
    base_directory = os.path.join(current_directory, 'experiments', 'kaggle-animals-10')
    total_files_processed = 0

    # for split in ['test', 'train', 'val']:
    for split in ['raw-img']:
        # for class_name in ['Cat', 'Dog']:
        for class_name in ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']:
            # delete 60% of the files in each class
            files = os.listdir(os.path.join(base_directory, split, class_name))
            print(f"Files start in {split}/{class_name}: {len(files)}")
            files_to_delete = int(len(files) - 1000)
            files_to_delete = files[:files_to_delete]
            for file in files_to_delete:
                os.remove(os.path.join(base_directory, split, class_name, file))
            
            # check if the files are deleted
            files = os.listdir(os.path.join(base_directory, split, class_name))
            print(f"Files remaining in {split}/{class_name}: {len(files)}")

            

