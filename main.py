import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.fft import dctn, idctn


def compress_image(image_path, quality_=50, block_size_=4):
    if is_gray(image_path):
        print('grayscale')
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        cv2.imshow('Original Image', img)
        cv2.waitKey(0)
    else:
        print('color')
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img is None:
        print("Error in reading the image.")
        exit()

    height, width = img.shape
    compressed_img = np.zeros_like(img)

    for i in range(0, height, block_size_):
        for j in range(0, width, block_size_):
            # potrzebny algorytm na wygladzanie miedzy blokami, to raczej nie dziala
            # Smooth along horizontal block boundaries
            block = img[i:i + block_size_, j:j + block_size_]
            block = np.float32(block)
            dct_block = dctn(block, norm='ortho')
            dct_block = np.int32(dct_block)
            if block_size_ == 4:
                quantization_table = [[16, 14, 12, 10],
                                      [14, 12, 10, 8],
                                      [12, 10, 8, 6],
                                      [10, 8, 6, 4]]
            else:
                quantization_table = [[16, 16, 16, 16, 16, 16, 16, 16],
                                      [16, 16, 16, 16, 16, 16, 16, 16],
                                      [16, 16, 16, 16, 16, 16, 16, 16],
                                      [16, 16, 16, 16, 16, 16, 16, 16],
                                      [16, 16, 16, 16, 16, 16, 16, 16],
                                      [16, 16, 16, 16, 16, 16, 16, 16],
                                      [16, 16, 16, 16, 16, 16, 16, 16],
                                      [16, 16, 16, 16, 16, 16, 16, 16]]

            dct_block_quantized = np.divide(dct_block, quantization_table)
            compressed_block = idctn(dct_block_quantized, norm='ortho')
            compressed_block = np.multiply(compressed_block, quantization_table)
            compressed_block = np.clip(compressed_block, 0, 255)
            compressed_img[i:i + block_size_, j:j + block_size_] = np.uint8(compressed_block)
    return compressed_img


def is_gray(img_path):
    img = cv2.imread(img_path)
    print(img.shape)
    if len(img.shape) < 3:
        return True
    if img.shape[2] == 1:
        return True
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
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
    input_image_path = 'baboon.ascii.pgm'
    output_image_path = 'baboon_comp.pgm'
    quality = 10
    block_size = 8
    compressed_image = compress_image(input_image_path, quality, block_size)
    save_pgm(output_image_path, compressed_image)

    print("Image compressed and saved successfully.")

    img = cv2.imread(input_image_path)
    cv2.imshow('Original Image', img)
    cv2.imshow('Compressed Image', compressed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
