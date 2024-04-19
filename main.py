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
    return scaled_table


def create_zig_zag_pattern(block_size_=8):
    zz_pattern = []
    x_idx, y_idx = 0, 0
    direction_flag = 1

    for i in range(block_size_ ** 2):
        zz_pattern.append((x_idx, y_idx))

        if direction_flag == 1:
            if y_idx == block_size_ - 1:
                x_idx += 1
                direction_flag = -1
            elif x_idx == 0:
                y_idx += 1
                direction_flag = -1
            else:
                x_idx -= 1
                y_idx += 1
        else:
            if x_idx == block_size_ - 1:
                y_idx += 1
                direction_flag = 1
            elif y_idx == 0:
                x_idx += 1
                direction_flag = 1
            else:
                x_idx += 1
                y_idx -= 1

    return zz_pattern


def zigzag_transform(block, zz_pattern):
    block_size_ = int(np.sqrt(len(zz_pattern)))
    zigzag_block = np.zeros((block_size_, block_size_), dtype=np.int32)

    for i, (y, x) in enumerate(zz_pattern):
        zigzag_block[y, x] = block[i]

    return zigzag_block


def reverse_zigzag_transform(zigzag_block, zz_pattern):
    block_size_ = int(np.sqrt(len(zz_pattern)))
    block = np.zeros((block_size_, block_size_), dtype=np.int32)
    block = block.flatten()
    x, y = zip(*zz_pattern)
    key = [0] * (len(zz_pattern))
    for i in range(len(zz_pattern)):
        key[i] = x[i] * block_size_ + y[i]
    for i in range(len(key)):
        block[i] = zigzag_block[key[i]]
    return block.reshape((block_size_, block_size_))


def run_length_encode(zz_img_list):
    run_length_encoded_list = []
    i = 0
    while i < len(zz_img_list):
        count = 1
        while i + 1 < len(zz_img_list) and zz_img_list[i] == zz_img_list[i + 1]:
            i += 1
            count += 1
        run_length_encoded_list.append(count)
        run_length_encoded_list.append(zz_img_list[i])
        i += 1

    return run_length_encoded_list


def run_length_decode(run_length_encoded_list):
    zz_img_list = []
    i = 0
    while i < len(run_length_encoded_list):
        count = run_length_encoded_list[i]
        pixel_value = run_length_encoded_list[i + 1]
        zz_img_list.extend([pixel_value] * count)
        i += 2
    return zz_img_list


def compress_image(image_path, quality_=50, block_size_=8):
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

    pad_height = (block_size_ - height % block_size_) % block_size_
    pad_width = (block_size_ - width % block_size_) % block_size_
    image = np.pad(image, ((0, pad_height), (0, pad_width)), mode='constant')
    altered_height, altered_width = image.shape

    compressed_img = np.zeros_like(image)
    base_table = load_and_slice_quantization_table('extended_table.txt', block_size_)
    quantization_table = create_quantization_table(quality_, base_table)

    zz_pattern = create_zig_zag_pattern(block_size_)
    zz_img_list = []
    run_length_list = []
    for i in range(0, altered_height, block_size_):
        for j in range(0, altered_width, block_size_):
            block = image[i:i + block_size_, j:j + block_size_]
            block = np.float64(block)
            block = block - 128
            block = dctn(block, norm='ortho')
            block = np.round(np.divide(block, quantization_table))

            zigzag_block = zigzag_transform(block.flatten(), zz_pattern)  # Zigzag transform
            zz_img_list.extend(zigzag_block.flatten())

    run_length_list = run_length_encode(zz_img_list)  # rl encoded full image
    decoded_run_length_list = run_length_decode(run_length_list)  # rl decoded full image

    idx = 0
    for i in range(0, altered_height, block_size_):
        for j in range(0, altered_width, block_size_):
            block_data = decoded_run_length_list[idx:idx + block_size_ ** 2]
            block = reverse_zigzag_transform(block_data, zz_pattern)
            idx += block_size_ ** 2
            block = np.multiply(block, quantization_table)
            block = idctn(block, norm='ortho')
            block = block + 128
            block = np.clip(block, 0, 255)
            compressed_img[i:i + block_size_, j:j + block_size_] = block

    print("first zig_zag block: " + str(zz_img_list[:block_size_ ** 2]))
    print("first decoded zig_zag block: " + str(decoded_run_length_list[:64]))
    print("zig_zag length: " + str(len(zz_img_list)))
    print("rl encoding length: " + str(len(run_length_list)))
    print("rl decoded length:" + str(len(decoded_run_length_list)))
    print(check_same_elements(zz_img_list, decoded_run_length_list))
    # huffman encoding to do
    # cv2.imshow("comp", compressed_img)
    # cv2.waitKey(0)
    compressed_img = compressed_img[:altered_height - pad_height, :altered_width - pad_width]
    return compressed_img


def check_same_elements(list1, list2):
    # Check if lists have the same length
    if len(list1) != len(list2):
        return "Lists are not the same"

    # Iterate through both lists simultaneously
    for elem1, elem2 in zip(list1, list2):
        # If elements are not equal, return False
        if elem1 != elem2:
            return "Lists are not the same"

    # If all elements are equal, return True
    return "Lists are the same"


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
    input_image_path = 'lena.pgm'
    output_image_path = 'lena_comp.pgm'
    quality = 50
    block_size = 8

    compressed_image = compress_image(input_image_path, quality, block_size)

    save_pgm(output_image_path, compressed_image)
    print("Image compressed and saved successfully.")
    original_size = round(os.path.getsize(input_image_path) / 1024)
    compressed_size = round(os.path.getsize(output_image_path) / 1024)
    print("Original size: " + str(original_size) + "kb")
    print("Compressed size: " + str(compressed_size) + "kb")
    print("Saved " + str(original_size - compressed_size) + "kb")
    img = cv2.imread(input_image_path)
    cv2.imshow('Original Image', cv2.resize(img, (600, 600)))
    cv2.imshow('Compressed Image', cv2.resize(compressed_image, (600, 600)))
    mse_value = helpers.calculate_mse(input_image_path, output_image_path)
    print("Mean Squared Error (MSE) between original and compressed images:", mse_value)
    psnr_value = helpers.calculate_psnr(input_image_path, output_image_path)
    print("Peak Signal-to-Noise Ratio (PSNR) between original and compressed images:", psnr_value)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
