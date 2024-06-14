import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dctn, idctn

import helpers

# File path to your CSV file
csv_file_path = 'lena.csv'

# Block sizes to be plotted
values = [10, 50, 80, 90, 100]

# Data storage
data = {
    'Quality': [],
    'Block Size': [],
    'Ydata': []
}

# Read the CSV file
with open(csv_file_path, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        data['Quality'].append(int(row['Quality']))
        data['Block Size'].append(int(row['Block Size']))
        data['Ydata'].append(float(row['Avg Opening Time']))


# Function to plot data for specific block sizes
def plot_data(data, values):
    plt.figure(figsize=(11.69, 8.27))
    for value in values:
        x_values = [data['Block Size'][i] for i in range(len(data['Block Size'])) if data['Quality'][i] == value]
        y_values = [data['Ydata'][i] for i in range(len(data['Ydata'])) if
                    data['Quality'][i] == value]

        plt.plot(x_values, y_values, linestyle='--', label=f'Quality {value}')

    plt.xlabel('Block Size')
    plt.ylabel('Avg Opening Time [s]')
    plt.title('Avg Opening Time vs Block Size for Different Quality Values (lena.isa)')
    plt.legend()
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.tight_layout()  # Adjust the layout to minimize empty space
    plt.show()


# Call the function to plot data
# plot_data(data, values)
import compressor

image = np.array([[10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
                  [15, 20, 25, 30, 35, 40, 45, 50, 55, 65],
                  [20, 25, 30, 35, 40, 45, 50, 55, 65, 70],
                  [25, 30, 35, 40, 45, 50, 55, 65, 70, 75],
                  [30, 35, 40, 45, 50, 55, 65, 70, 75, 80],
                  [35, 40, 45, 50, 55, 65, 70, 75, 80, 85],
                  [40, 45, 50, 55, 65, 70, 75, 80, 85, 90],
                  [45, 50, 55, 65, 70, 75, 80, 85, 90, 95],
                  [50, 55, 65, 70, 75, 80, 85, 90, 95, 100],
                  [55, 65, 70, 75, 80, 85, 90, 95, 100, 105]
                  ], dtype=np.int32)

base = helpers.create_base_table(10)
# print(base)
scaled = compressor.create_quantization_table(75, base)
# print(scaled)

block = np.float64(image)
block = block - 128
# print(block)
block = dctn(block, norm='ortho')
# print(block)
block = np.round(np.divide(block, scaled)).astype(int)
# print(block)

zz_pattern = compressor.create_zig_zag_pattern(10)

zz_list = compressor.zigzag_transform(block, zz_pattern)
# print(zz_list)

rle = compressor.run_length_encode(zz_list)
print(rle)

from collections import Counter

# Count frequency of numbers
#frequency = Counter(rle)
# Sort the frequency dictionary by the frequency of occurrences
#sorted_frequency = sorted(frequency.items(), key=lambda item: item[1], reverse=True)
# Print the frequency of each number
#for num, freq in sorted_frequency:
#   print(f"Number: {num}, Frequency: {freq}")

import Huffman
tree = Huffman.build_huffman_tree(rle)
codes = Huffman.generate_huffman_codes(tree)
print(codes)
huff_encoded = Huffman.huffman_encode(rle, codes)
print(huff_encoded)

binary = Huffman.binary_string_to_bytes(huff_encoded)
print(binary)

string = Huffman.bytes_to_binary_string(binary)
print(string)

huff_decoded = Huffman.huffman_decode(string, codes)
print(huff_decoded)

stringu = Huffman.binary_string_to_bytes(string)
print(stringu)

