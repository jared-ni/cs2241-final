import numpy as np
from PIL import Image
import os

# arr = np.array(Image.open('imagenet-sample-images/n01443537_goldfish.JPEG'))
# arr = arr // 64 * 64
# Image.fromarray(arr).save('out.png')


arr = np.array(Image.open('imagenet-sample-images/n01443537_goldfish.JPEG'))
height, width, _ = arr.shape
arr = arr // 64 * 64
red = arr[:, :, 0]
green = out0 = arr[:, :, 1]
blue = arr[:, :, 2]
red3 = np.zeros_like(arr)
green3 = np.zeros_like(arr)
blue3 = np.zeros_like(arr)
red3[:, :, 0] = red
green3[:, :, 1] = green
blue3[:, :, 2] = blue
Image.fromarray(red3).save('red.png')
Image.fromarray(green3).save('green.png')
Image.fromarray(blue3).save('blue.png')
