import os
import time
import csv
import compressor
import helpers

input_filepath = 'images\\samples\\lena.pgm'
output_filepath = 'test_image.isa'

test_image = compressor.open_image(input_filepath)

csv_filename = 'lena.csv'

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ['Quality', 'Block Size', 'Avg Compression Time', 'Avg Saving Time', 'Avg Opening Time', 'MSE', 'PSNR',
         'Compressed Size'])

for quality in range(1, 101):
    for block_size in range(1, 65):
        compression_time, saving_time = compressor.save_isa_testing(output_filepath, test_image, quality,
                                                                    block_size)
        opened_dct_time_start = time.time()
        opened_dct_image = compressor.open_image(output_filepath)
        opened_dct_time_end = time.time()
        opening_time = opened_dct_time_end - opened_dct_time_start
        mse = helpers.calculate_mse(test_image, opened_dct_image)
        psnr = helpers.calculate_psnr(test_image, opened_dct_image)
        size = os.path.getsize(output_filepath)/1024
        print(f"Progress: Quality {quality}, Block Size {block_size}")

        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                [quality, block_size, compression_time, saving_time, opening_time, mse, psnr, size])
