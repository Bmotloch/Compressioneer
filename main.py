import os
import cv2
import numpy as np
from scipy.fft import dctn, idctn

import helpers

def load_and_slice_quantization_table(file_path, slice_size):
    quantization_table_32x32 = np.loadtxt(file_path, dtype=int)

    if quantization_table_32x32.shape != (32, 32):
        raise ValueError("Loaded quantization table must be 32x32")

    sliced_table = quantization_table_32x32[:slice_size, :slice_size]
    return sliced_table


def create_quantization_table(quality_factor, base_table_):
    if quality_factor < 50:
        scaling_factor = 5000 / quality_factor
    else:
        scaling_factor = 200 - (quality_factor * 2)

    scaled_table = np.clip(np.int32(np.floor((base_table_ * scaling_factor + 50) / 100)), 1, 255)
    # print(scaled_table)
    return scaled_table


def compress_image(image_path, quality_=50, block_size_=4):
    # potrzebny algorytm na wygladzanie miedzy blokami
    if is_gray(image_path):
        print('grayscale')
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        print('color')
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        print("Error in reading the image.")
        exit()

    height, width = image.shape

    # Calculate the padding needed
    pad_height = (block_size_ - height % block_size_) % block_size_
    pad_width = (block_size_ - width % block_size_) % block_size_

    # Pad the image
    image = np.pad(image, ((0, pad_height), (0, pad_width)), mode='constant')

    # Update height and width after padding
    height, width = image.shape

    compressed_img = np.zeros_like(image)
    # for now every table is based on standard 8x8 jpeg q=50,
    # will be created as one big 32x32 extended jpeg 8x8 array and sliced as user defines
    # edit - based on making the 8x8 symmetrical and extending the table by interpolation, kinda works
    """
    if block_size_ == 4:
        base_table = np.array([[16, 11, 10, 16],
                               [12, 12, 14, 19],
                               [14, 13, 16, 24],
                               [14, 17, 22, 29]])
    elif block_size_ == 5:
        base_table = np.array([[16, 11, 10, 16, 24],
                               [12, 12, 14, 19, 26],
                               [14, 13, 16, 24, 40],
                               [14, 17, 22, 29, 51],
                               [18, 22, 37, 56, 68]])
    elif block_size_ == 6:
        base_table = np.array([[16, 11, 10, 16, 24, 40],
                               [12, 12, 14, 19, 26, 58],
                               [14, 13, 16, 24, 40, 57],
                               [14, 17, 22, 29, 51, 87],
                               [18, 22, 37, 56, 68, 109],
                               [24, 35, 55, 64, 81, 104]])
    elif block_size_ == 8:
        base_table = np.array([[16, 11, 12, 15, 21, 32, 50, 66],
                               [11, 12, 13, 18, 24, 46, 62, 73],
                               [12, 13, 16, 23, 38, 56, 73, 75],
                               [15, 18, 23, 29, 53, 75, 83, 80],
                               [21, 24, 38, 53, 68, 95, 103, 94],
                               [32, 46, 56, 75, 95, 104, 117, 96],
                               [50, 62, 73, 83, 103, 117, 120, 102],
                               [66, 73, 75, 80, 94, 96, 102, 99]])
    else:
        base_table = np.array([[16, 11, 10, 16],
                               [12, 12, 14, 19],
                               [14, 13, 16, 24],
                               [14, 17, 22, 29]])
    """
    base_table = load_and_slice_quantization_table('extended_table.txt', block_size_)
    quantization_table = create_quantization_table(quality_, base_table)

    for i in range(0, height, block_size_):
        for j in range(0, width, block_size_):
            block = image[i:i + block_size_, j:j + block_size_]
            block = np.float32(block)
            block = block - 127
            block = dctn(block, norm='ortho')
            block = np.divide(block, quantization_table)
            block = np.int32(block)

            compressed_block = np.multiply(block, quantization_table)
            compressed_block = idctn(compressed_block, norm='ortho')
            compressed_block = compressed_block + 127
            compressed_block = np.clip(compressed_block, 0, 255)
            compressed_img[i:i + block_size_, j:j + block_size_] = np.uint8(compressed_block)

    # Remove padding before returning the compressed image
    compressed_img = compressed_img[:height - pad_height, :width - pad_width]
    return compressed_img


def is_gray(img_path):
    image = cv2.imread(img_path)
    print(image.shape)
    if len(image.shape) < 3:
        return True
    if image.shape[2] == 1:
        return True
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    if (b == g).all() and (b == r).all():
        return True
    return False


def save_pgm(filename, image):
    with open(filename, 'w') as f:
        f.write("P2\n")
        f.write("# ISA certified\n")
        f.write("{} {}\n".format(image.shape[1], image.shape[0]))
        f.write("255\n")

        for row in image:
            for pixel in row:
                f.write("{} ".format(int(pixel)))
            f.write("\n")


if __name__ == "__main__":
    input_image_path = 'baboon.pgm'
    output_image_path = 'baboon_comp.pgm'
    quality = 100
    block_size = 32

    compressed_image = compress_image(input_image_path, quality, block_size)
    save_pgm(output_image_path, compressed_image)
    print("Image compressed and saved successfully.")
    original_size = round(os.path.getsize(input_image_path) / 1024)
    compressed_size = round(os.path.getsize(output_image_path) / 1024)
    print("Original size: " + str(original_size) + "kb")
    print("Compressed size: " + str(compressed_size) + "kb")
    print("Saved "+ str(original_size-compressed_size) + "kb")
    img = cv2.imread(input_image_path)
    cv2.imshow('Original Image', img)
    cv2.imshow('Compressed Image', compressed_image)
    mse_value = helpers.calculate_mse(input_image_path, output_image_path)
    print("Mean Squared Error (MSE) between original and compressed images:", mse_value)
    psnr_value = helpers.calculate_psnr(input_image_path, output_image_path)
    print("Peak Signal-to-Noise Ratio (PSNR) between original and compressed images:", psnr_value)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
