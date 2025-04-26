from PIL import Image
import os


def resize_image(image_path, save_path, dims, padding):
    # Open the image
    img = Image.open(image_path)
    
    # Convert to RGB to ensure 3 channels
    img = img.convert('RGB')

    if padding:
        # Find the new size keeping aspect ratio
        img.thumbnail(dims, Image.LANCZOS)
        # Create a new background
        new_img = Image.new('RGB', dims, (0, 0, 0))
        # Paste the resized image onto the center of the background
        paste_x = (dims[0] - img.width) // 2
        paste_y = (dims[1] - img.height) // 2
        new_img.paste(img, (paste_x, paste_y))

    else:
        # Allow image to be distorted to fill the space
        new_img = img.resize(dims, Image.LANCZOS)
    
    # Save
    new_img.save(save_path)


in_dir_name = 'imagenet-sample-images'
out_dir_name = 'imagenet-sample-images-resized'
dims = (256, 256)
padding = True

for file_name in os.listdir(in_dir_name):
    image_path = f'{in_dir_name}/{file_name}'
    save_path = f'{out_dir_name}/{file_name}'
    resize_image(image_path, save_path, dims, padding)
