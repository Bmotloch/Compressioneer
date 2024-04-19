import heapq
from collections import Counter


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
    decoded_data = []
    current_code = ""
    for bit in encoded_data:
        current_code += bit
        for symbol, code in huffman_codes.items():
            if current_code == code:
                decoded_data.append(int(symbol))
                current_code = ""
                break
    return decoded_data


def write_isa_file(filename, encoded_data, huffman_codes, quality, block_size, height, width):
    with open(filename, 'w') as f:
        f.write(f"{height}\n")
        f.write(f"{width}\n")
        f.write(f"{quality}\n")
        f.write(f"{block_size}\n")

        for symbol, code in huffman_codes.items():
            f.write(f"{symbol}:{code}\n")

        f.write("---\n")
        f.write(encoded_data)


def read_isa_file(filename):
    huffman_codes = {}
    encoded_data = ""
    quality = 0
    block_size = 0
    height = 0
    width = 0
    with open(filename, 'r') as f:
        height = int(f.readline().strip())
        width = int(f.readline().strip())
        quality = int(f.readline().strip())
        block_size = int(f.readline().strip())

        line = f.readline().strip()
        while line != "---":
            symbol, code = line.split(":")
            huffman_codes[symbol] = code
            line = f.readline().strip()

        encoded_data = f.read().strip()

    return encoded_data, huffman_codes, quality, block_size, height, width


def save_isa(filename, run_length_encoded_data, quality, block_size, height, width):
    tree = build_huffman_tree(run_length_encoded_data)
    codes = generate_huffman_codes(tree)
    huff_encoded = huffman_encode(run_length_encoded_data, codes)
    write_isa_file(filename, huff_encoded, codes, quality, block_size, height, width)  # file written
