import numpy as np
from PIL import Image


class Compressor:
    def __init__(self, channel_granularity, anchor_frequency, bloom_fpp):
         self.channel_granularity = channel_granularity
         self.quantization_factor = 256 // channel_granularity
         self.anchor_frequency = anchor_frequency
         self.bloom_fpp = bloom_fpp

    def compress_image(self, image_file, compressed_file):
        # Load image as NumPy array
        arr = np.array(Image.open(image_file))
        self.height, self.width = arr.shape[0], arr.shape[1]

        # Quantize
        arr //= self.quantization_factor

        # RGB images
        if len(arr.shape) == 3:
            red = arr[:, :, 0].flatten()
            green = arr[:, :, 1].flatten()
            blue = arr[:, :, 2].flatten()
            # red = self.delta_encoding(arr[:, :, 0].flatten())
            # green = self.delta_encoding(arr[:, :, 1].flatten())
            # blue = self.delta_encoding(arr[:, :, 2].flatten())
            # Regress green and blue channels on red
            green_res, green_beta0, green_beta1 = self.lin_reg(red, green)
            blue_res, blue_beta0, blue_beta1 = self.lin_reg(red, blue)
            self.compress_arr(red)
            self.compress_arr(green)
            self.compress_arr(green_res)

        # Grayscale images
        else:
            arr = arr.flatten()
            # self.compress_arr(arr)

    def compress_arr(self, arr: np.ndarray):
        arr = self.delta_encoding(arr)
        # print(arr)
        print(f'{round((arr!=0).mean()*100, 2)}%', end='\t')
        # print(f'{self.quantization_factor*np.abs(arr).mean():.2f}', end='\t')
        # out = np.abs(arr).astype(np.uint8).reshape(self.height, self.width)
        # print(f'{round((out>0).mean()*100)}%', end='\t')

    def lin_reg(self, x: np.ndarray, y: np.ndarray):
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        xbar = x.mean()
        ybar = y.mean()
        beta1: np.float64 = ((x-xbar)*(y-ybar)).sum() / ((x-xbar)**2).sum()
        beta0: np.float64 = ybar - beta1 * xbar
        # yhat = np.round(beta0 + beta1 * x).astype(int)
        yhat = beta0 + beta1 * x
        res = (y - yhat).astype(np.int8)
        return res, beta0, beta1

    def delta_encoding(self, arr: np.ndarray):
        enc = np.ndarray(arr.shape, np.int8)
        enc[1:] = np.diff(arr.astype(np.int8))
        enc[::self.anchor_frequency] = arr[::self.anchor_frequency]
        return enc


import os
dir_name = 'imagenet-sample-images'
compressor = Compressor(channel_granularity=4, anchor_frequency=1000000000000000000, bloom_fpp=0.01)
for file_name in os.listdir(dir_name):
    compressor.compress_image(f'{dir_name}/{file_name}', None)
    print()


def delta_encoding(arr):
    return np.concatenate(([arr[0]], np.diff(arr.astype(np.int16))))


# import os
# dir_name = 'imagenet-sample-images'
# for file_name in os.listdir(dir_name):
#     arr = np.array(Image.open(f'{dir_name}/{file_name}'))
#     if len(arr.shape) == 2:
#         continue
#     height, width, _ = arr.shape
#     arr //= 64
#     red = arr[:, :, 0]
#     green = out0 = arr[:, :, 1]
#     blue = arr[:, :, 2]
#     # Image.fromarray(red).save('red.png')
#     # Image.fromarray(green).save('green.png')
#     # Image.fromarray(blue).save('blue.png')
#     x = red.flatten().astype(np.float64)
#     y = green.flatten().astype(np.float64)
#     xbar = x.mean()
#     ybar = y.mean()
#     beta1 = ((x-xbar)*(y-ybar)).sum() / ((x-xbar)**2).sum()
#     beta0 = ybar - beta1 * xbar
#     green_res = y - (beta0 + beta1 * x)
#     print(y)
#     # print(green_res)
#     continue
#     # out1 = np.abs(green_res).astype(np.uint8).reshape(height, width)
#     # Image.fromarray(out1).save('green_res.png')
#     green_res_diff = delta_encoding(green_res.astype(int))
#     out2 = np.abs(green_res_diff).astype(np.uint8).reshape(height, width)
#     out1 = np.abs(delta_encoding(y.astype(int))).astype(np.uint8).reshape(height, width)
#     # Image.fromarray(out2).save('green_res_diff.png')
#     print(f'{out0.mean():.2f}\t{out1.mean():.2f}\t{round((out1>0).mean()*100)}%\t{out2.mean():.2f}\t{round((out2>0).mean()*100)}%\t{file_name}')
