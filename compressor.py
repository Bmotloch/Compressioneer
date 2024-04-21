import cv2
import numpy as np
from scipy.fft import dctn, idctn
import Huffman


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


def reverse_zigzag_transform(zigzag_block, zz_pattern, key):
    block_size_ = int(np.sqrt(len(zz_pattern)))
    block = np.zeros((block_size_, block_size_), dtype=np.int32)
    block = block.flatten()
    for i in range(len(key)):
        block[i] = zigzag_block[key[i]]
    return block.reshape((block_size_, block_size_))


def create_zigzag_key(zz_pattern, block_size):
    x, y = zip(*zz_pattern)
    key = [0] * (len(zz_pattern))
    for i in range(len(zz_pattern)):
        key[i] = x[i] * block_size + y[i]
    return key


def delta_encode(data):
    delta_encoded = [data[0]]
    for i in range(1, len(data)):
        delta = data[i] - data[i - 1]
        delta_encoded.append(delta)
    return delta_encoded


def decode_delta(delta_encoded):
    original_data = [delta_encoded[0]]
    for i in range(1, len(delta_encoded)):
        original_value = original_data[i - 1] + delta_encoded[i]
        original_data.append(original_value)
    return original_data


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


def perform_dct(input_image_path, quality_=50, block_size_=8):
    if input_image_path.endswith('.isa'):
        image = decompress_isa(input_image_path)
    elif input_image_path.endswith('.pgm'):
        image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # temporary

    height_, width_ = image.shape

    pad_height = (block_size_ - height_ % block_size_) % block_size_
    pad_width = (block_size_ - width_ % block_size_) % block_size_
    padded_height = height_ + pad_height
    padded_width = width_ + pad_width

    image = np.pad(image, ((0, pad_height), (0, pad_width)), mode='constant')
    dct_image = np.zeros_like(image)

    base_table = load_and_slice_quantization_table('extended_table.txt', block_size_)
    quantization_table = create_quantization_table(quality_, base_table)
    for i in range(0, padded_height, block_size_):
        for j in range(0, padded_width, block_size_):
            block = image[i:i + block_size_, j:j + block_size_]
            block = np.float64(block)
            block = block - 128
            block = dctn(block, norm='ortho')
            block = np.round(np.divide(block, quantization_table))
            block = np.multiply(block, quantization_table)
            block = idctn(block, norm='ortho')
            block = block + 128
            block = np.clip(block, 0, 255)
            dct_image[i:i + block_size_, j:j + block_size_] = block

    dct_image = np.uint8(dct_image[:height_, :width_])
    return dct_image


def decompress_isa(encoded_image_path):
    encoded_isa_data, isa_codes, quality_, block_size_, height_, width_, rl_flag_, delta_flag_ = Huffman.read_isa_file(
        encoded_image_path)
    decoded_isa_data = Huffman.huffman_decode(encoded_isa_data, isa_codes)
    print(f"rl_flag:{rl_flag_}\n"  # seems to prefer only run length
          f"delta_flag:{delta_flag_}")
    if rl_flag_ == 1 and delta_flag_ == 1:
        decoded_run_length_list = run_length_decode(decoded_isa_data)
        dct_data = decode_delta(decoded_run_length_list)
    elif rl_flag_ == 1 and delta_flag_ == 0:
        dct_data = run_length_decode(decoded_isa_data)
    else:
        dct_data = decoded_isa_data
    pad_height = (block_size_ - height_ % block_size_) % block_size_
    pad_width = (block_size_ - width_ % block_size_) % block_size_
    padded_height = height_ + pad_height
    padded_width = width_ + pad_width

    compressed_img = np.zeros((padded_height, padded_width))

    base_table = load_and_slice_quantization_table('extended_table.txt', block_size_)
    quantization_table = create_quantization_table(quality_, base_table)
    zz_pattern = create_zig_zag_pattern(block_size_)
    zz_key = create_zigzag_key(zz_pattern, block_size_)
    idx = 0
    for i in range(0, padded_height, block_size_):
        for j in range(0, padded_width, block_size_):
            block_data = dct_data[idx:idx + block_size_ ** 2]
            block = reverse_zigzag_transform(block_data, zz_pattern, zz_key)
            idx += block_size_ ** 2
            block = np.multiply(block, quantization_table)
            block = idctn(block, norm='ortho')
            block = block + 128
            block = np.clip(block, 0, 255)
            compressed_img[i:i + block_size_, j:j + block_size_] = block

    decompressed_img = np.uint8(compressed_img[:height_, :width_])
    return decompressed_img


def save_image(dct_image, output_image_path, compressed_quality, compressed_block_size):
    if output_image_path.endswith('.isa'):
        save_isa(output_image_path, dct_image, compressed_quality, compressed_block_size)
    elif output_image_path.endswith('.pgm'):
        save_pgm(output_image_path, dct_image)
    else:
        cv2.imwrite(output_image_path, dct_image)


def save_isa(output_image_path, dct_image, compressed_quality, compressed_block_size):
    height, width = dct_image.shape

    # Calculate padding
    pad_height = (compressed_block_size - height % compressed_block_size) % compressed_block_size
    pad_width = (compressed_block_size - width % compressed_block_size) % compressed_block_size
    padded_height = height + pad_height
    padded_width = width + pad_width

    base_table = load_and_slice_quantization_table('extended_table.txt', compressed_block_size)
    quantization_table = create_quantization_table(compressed_quality, base_table)
    zz_pattern = create_zig_zag_pattern(compressed_block_size)
    zz_img_list = []
    delta_flag = 1
    rl_flag = 1
    dct_image = np.pad(dct_image, ((0, pad_height), (0, pad_width)), mode='constant')
    for i in range(0, padded_height, compressed_block_size):
        for j in range(0, padded_width, compressed_block_size):
            block = dct_image[i:i + compressed_block_size, j:j + compressed_block_size]
            block = np.float64(block)
            block = block - 128
            block = dctn(block, norm='ortho')
            block = np.round(np.divide(block, quantization_table))
            zigzag_block = zigzag_transform(block.flatten(), zz_pattern)
            zz_img_list.extend(zigzag_block.flatten())

    no_encoding_size = len(zz_img_list)

    run_length_list = run_length_encode(zz_img_list)
    run_length_size = len(run_length_list)

    delta_encoded_list = delta_encode(zz_img_list)
    delta_run_length_list = run_length_encode(delta_encoded_list)
    delta_run_length_size = len(delta_run_length_list)

    print(f"no encoding length:{no_encoding_size}\n"
          f"rl only encoding length:{run_length_size}\n"
          f"rl+delta only encoding length:{delta_run_length_size}")

    if no_encoding_size <= run_length_size and no_encoding_size <= delta_run_length_size:
        Huffman.save_isa(output_image_path, zz_img_list, compressed_quality, compressed_block_size, height, width, 0, 0)
    elif run_length_size <= delta_run_length_size:
        Huffman.save_isa(output_image_path, run_length_list, compressed_quality, compressed_block_size, height, width,
                         1, 0)
    else:
        Huffman.save_isa(output_image_path, delta_run_length_list, compressed_quality, compressed_block_size, height,
                         width, 1, 1)


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


def open_image(image_path):
    if image_path.endswith('.isa'):
        image = decompress_isa(image_path)
    elif image_path.endswith('.pgm'):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # temporary
    return image
