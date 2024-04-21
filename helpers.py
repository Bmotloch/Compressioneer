import numpy as np


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


def make_symmetrical_quantization_table():
    quantization_table = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                   [12, 12, 14, 19, 26, 58, 60, 55],
                                   [14, 13, 16, 24, 40, 57, 69, 56],
                                   [14, 17, 22, 29, 51, 87, 80, 62],
                                   [18, 22, 37, 56, 68, 109, 103, 77],
                                   [24, 35, 55, 64, 81, 104, 113, 92],
                                   [49, 64, 78, 87, 103, 121, 120, 101],
                                   [72, 92, 95, 98, 112, 100, 103, 99]])

    symmetrical_table = (quantization_table + quantization_table.T) // 2

    return symmetrical_table


def interpolate(matrix, new_size):
    height, width = matrix.shape
    new_matrix = np.zeros((new_size, new_size), dtype=np.uint8)

    if new_size == 1:
        return np.ones((1, 1), dtype=np.uint8) * 16

    x_ratio = float(width - 1) / (new_size - 1)
    y_ratio = float(height - 1) / (new_size - 1)

    for i in range(new_size):
        for j in range(new_size):
            x = int(x_ratio * i)
            y = int(y_ratio * j)
            x_diff = (x_ratio * i) - x
            y_diff = (y_ratio * j) - y

            # Handling boundary conditions
            if x == width - 1:
                x = width - 2
            if y == height - 1:
                y = height - 2

            # Bilinear interpolation
            interpolated_value = (1 - x_diff) * (1 - y_diff) * matrix[y][x] + \
                                 x_diff * (1 - y_diff) * matrix[y][x + 1] + \
                                 (1 - x_diff) * y_diff * matrix[y + 1][x] + \
                                 x_diff * y_diff * matrix[y + 1][x + 1]

            new_matrix[j][i] = interpolated_value

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
