import numpy as np


def interpolate(matrix, new_size):
    height, width = matrix.shape
    new_matrix = np.zeros((new_size, new_size), dtype=np.int32)

    if new_size == 1:
        return np.ones((1, 1), dtype=np.int32) * 16

    x_ratio = float(width - 1) / (new_size - 1)
    y_ratio = float(height - 1) / (new_size - 1)

    for i in range(new_size):
        for j in range(new_size):
            x = int(x_ratio * i)
            y = int(y_ratio * j)
            x_diff = (x_ratio * i) - x
            y_diff = (y_ratio * j) - y

            if x == width - 1:
                x = width - 2
            if y == height - 1:
                y = height - 2

            interpolated_value = (1 - x_diff) * (1 - y_diff) * matrix[y][x] + \
                                 x_diff * (1 - y_diff) * matrix[y][x + 1] + \
                                 (1 - x_diff) * y_diff * matrix[y + 1][x] + \
                                 x_diff * y_diff * matrix[y + 1][x + 1]

            new_matrix[j][i] = int(interpolated_value)

    return new_matrix


def create_base_table(block_size):
    template_table = np.array([[16, 11, 12, 15, 21, 32, 50, 66],
                               [11, 12, 13, 18, 24, 46, 62, 73],
                               [12, 13, 16, 23, 38, 56, 73, 75],
                               [15, 18, 23, 29, 53, 75, 83, 80],
                               [21, 24, 38, 53, 68, 95, 103, 94],
                               [32, 46, 56, 75, 95, 104, 117, 96],
                               [50, 62, 73, 83, 103, 117, 120, 102],
                               [66, 73, 75, 80, 94, 96, 102, 99]])

    base_table = interpolate(template_table, block_size)

    return base_table


def calculate_mse(original_img, compressed_img):
    if original_img.shape != compressed_img.shape:
        raise ValueError("Original and compressed images must have the same shape")

    mse = np.mean((original_img - compressed_img) ** 2)
    return mse


def calculate_psnr(original_img, compressed_img):
    if original_img.shape != compressed_img.shape or original_img.dtype != compressed_img.dtype:
        raise ValueError("Original and compressed images must have the same shape and data type")

    mse = np.mean((original_img - compressed_img) ** 2)

    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0 if original_img.dtype == np.uint8 else 1.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr
