import numpy as np
import cv2


def extend_to_32x32(quantization_table):
    if quantization_table.shape != (8, 8):
        raise ValueError("Input quantization table must be 8x8")

    row_factor = 32 / 8
    col_factor = 32 / 8

    extended_table = np.zeros((32, 32), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            i_start = int(i * row_factor)
            i_end = int((i + 1) * row_factor)
            j_start = int(j * col_factor)
            j_end = int((j + 1) * col_factor)

            for x in range(i_start, i_end):
                for y in range(j_start, j_end):
                    frac_i = (x - i_start) / (i_end - i_start)
                    frac_j = (y - j_start) / (j_end - j_start)

                    if j == 7:
                        if i == 7:
                            value = quantization_table[i, j]
                        else:
                            value = (1 - frac_i) * quantization_table[i, j] + frac_i * quantization_table[i + 1, j]
                    elif i == 7:
                        value = (1 - frac_j) * quantization_table[i, j] + frac_j * quantization_table[i, j + 1]
                    else:
                        value = (1 - frac_i) * (1 - frac_j) * quantization_table[i, j] + \
                                (1 - frac_i) * frac_j * quantization_table[i, j + 1] + \
                                frac_i * (1 - frac_j) * quantization_table[i + 1, j] + \
                                frac_i * frac_j * quantization_table[i + 1, j + 1]

                    extended_table[x, y] = value

    return np.round(extended_table).astype(int)


def make_symmetrical_quantization_table(quantization_table):
    if quantization_table.shape != (8, 8):
        raise ValueError("Input quantization table must be 8x8")

    symmetrical_table = (quantization_table + quantization_table.T) // 2

    return symmetrical_table


def calculate_mse(input_image_path, output_image_path):
    # Ensure both images have the same shape
    original_img_ = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    compressed_img_ = cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE)
    if original_img_.shape != compressed_img_.shape:
        raise ValueError("Original and compressed images must have the same shape")

    # Calculate MSE
    mse = np.mean((original_img_ - compressed_img_) ** 2)
    return mse


def calculate_psnr(input_image_path, output_image_path):
    # Ensure both images have the same shape
    original_img_ = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    compressed_img_ = cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE)
    if original_img_.shape != compressed_img_.shape or original_img_.dtype != compressed_img_.dtype:
        raise ValueError("Original and compressed images must have the same shape and data type")
    mse = np.mean((original_img_ - compressed_img_) ** 2)

    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0 if original_img_.dtype == np.uint8 else 1.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
