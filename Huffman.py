import heapq
from collections import Counter
import pickle
import zlib


class Node:
    def __init__(self, freq, value=None):
        self.freq = freq
        self.value = value
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(data):
    freq_dict = Counter(data)
    priority_queue = [Node(freq, value) for value, freq in freq_dict.items()]
    heapq.heapify(priority_queue)

    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        merged = Node(left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(priority_queue, merged)

    return priority_queue[0]


def generate_huffman_codes(root):
    codes = {}

    def traverse(node, code):
        if node:
            if node.value is not None:
                codes[node.value] = code
            traverse(node.left, code + '0')
            traverse(node.right, code + '1')

    traverse(root, '')
    return codes


def huffman_encode(data, codes):
    encoded_data = ''.join(codes[char] for char in data)
    return encoded_data


def huffman_decode(encoded_data, huffman_codes):
    code_to_symbol = {code: int(symbol) for symbol, code in huffman_codes.items()}

    decoded_data = []
    current_code = []

    for bit in encoded_data:
        current_code.append(bit)
        current_code_str = ''.join(current_code)
        if current_code_str in code_to_symbol:
            decoded_data.append(code_to_symbol[current_code_str])
            current_code = []

    return decoded_data


def write_isa_file(filename, encoded_data, huffman_codes, quality, block_size, height, width, rl_flag):
    encoded_bytes = binary_string_to_bytes(encoded_data)

    data_to_write = {
        'encoded_data': encoded_bytes,
        'huffman_codes': huffman_codes,
        'quality': quality,
        'block_size': block_size,
        'height': height,
        'width': width,
        'rl_flag': rl_flag,
    }

    compressed_data = zlib.compress(pickle.dumps(data_to_write))

    with open(filename, 'wb') as f:
        f.write(compressed_data)


def binary_string_to_bytes(binary_string):
    padding_len = (8 - len(binary_string) % 8) % 8
    binary_string = f"{padding_len:08b}" + binary_string + '0' * padding_len
    return bytes(int(binary_string[i: i + 8], 2) for i in range(0, len(binary_string), 8))


def bytes_to_binary_string(byte_array):
    padding_len = byte_array[0]
    binary_string = ''.join(format(byte, '08b') for byte in byte_array[1:])
    return binary_string[:-padding_len] if padding_len > 0 else binary_string


def read_isa_file(filename):
    with open(filename, 'rb') as f:
        compressed_data = f.read()

    decompressed_data = zlib.decompress(compressed_data)
    data = pickle.loads(decompressed_data)
    encoded_data = bytes_to_binary_string(data['encoded_data'])

    return (
        encoded_data,
        data['huffman_codes'],
        data['quality'],
        data['block_size'],
        data['height'],
        data['width'],
        data['rl_flag'],
    )


def save_isa(filename, data, quality, block_size, height, width, rl_flag):
    tree = build_huffman_tree(data)
    codes = generate_huffman_codes(tree)
    huff_encoded = huffman_encode(data, codes)
    write_isa_file(filename, huff_encoded, codes, quality, block_size, height, width, rl_flag)
