import cv2
import numpy as np
from scipy.fft import dctn, idctn
from app import MainWindow
import sys
from PyQt5.QtWidgets import QApplication
import os
import helpers
import compressor

if __name__ == "__main__":
    """
    uncompressed_image = compressor.open_image('lena.isa')
    # decompressed_image = open_image(output_image_path)
    # original_size = round(os.path.getsize(input_image_path) / 1024)
    # compressed_size = round(os.path.getsize(output_image_path) / 1024)
    # print("Original size: " + str(original_size) + "kb")
    # print("Compressed size: " + str(compressed_size) + "kb")
    # print("Difference: " + str(original_size - compressed_size) + "kb")
    cv2.imshow('Uncompressed_image', cv2.resize(uncompressed_image, (600, 600)))
    # cv2.imshow('Compressed Image', cv2.resize(decompressed_image, (600, 600)))
    # mse_value = helpers.calculate_mse(uncompressed_image, decompressed_image)
    # print("Mean Squared Error (MSE) between original and compressed images:", mse_value)
    # psnr_value = helpers.calculate_psnr(uncompressed_image, decompressed_image)
    # print("Peak Signal-to-Noise Ratio (PSNR) between original and compressed images:", psnr_value)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

