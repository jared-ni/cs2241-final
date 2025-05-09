import numpy as np
from scipy import ndimage
from PIL import Image
import os

def get_outline(image_path, save_path_img, save_path_npy):
    # Load the image in grayscale
    image = np.array(Image.open(image_path))
    image = image.mean(axis=2)

    # Define Sobel kernels
    sobel_x_kernel = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

    sobel_y_kernel = np.array([[ 1,  2,  1],
                            [ 0,  0,  0],
                            [-1, -2, -1]])

    # Convolve image with Sobel kernels
    sobel_x = ndimage.convolve(image.astype(float), sobel_x_kernel)
    sobel_y = ndimage.convolve(image.astype(float), sobel_y_kernel)

    # Compute gradient magnitude
    sobel_magnitude = np.hypot(sobel_x, sobel_y)  # sqrt(x^2 + y^2)
    sobel_magnitude = sobel_magnitude / sobel_magnitude.max()  # Normalize
    sobel_magnitude = (np.round(sobel_magnitude * 3).astype(int) * 255 // 3).astype(np.uint8)

    Image.fromarray(sobel_magnitude).save(save_path_img)
    np.save(save_path_npy, sobel_magnitude)


in_dir_name = 'imagenet-sample-images-resized'
out_dir_name_img = 'imagenet-sample-images-outlines-img'
out_dir_name_npy = 'imagenet-sample-images-outlines-npy'

for file_name in os.listdir(in_dir_name):
    file_name_no_ext = file_name[:file_name.rfind('.')]
    image_path = f'{in_dir_name}/{file_name}'
    save_path_img = f'{out_dir_name_img}/{file_name_no_ext}.png'
    save_path_npy = f'{out_dir_name_npy}/{file_name_no_ext}.npy'
    get_outline(image_path, save_path_img, save_path_npy)
