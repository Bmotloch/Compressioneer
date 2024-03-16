import cv2
import numpy as np


def compress_image(image_path, quality_=20, block_size_=8):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error in reading the image.")
        exit()

    height, width = img.shape
    compressed_img = np.zeros_like(img)

    for i in range(0, height, block_size_):
        for j in range(0, width, block_size_):
            block = img[i:i + block_size_, j:j + block_size_]
            dct_block = cv2.dct(np.float32(block) / 255.0)
            dct_block_quantized = np.round(dct_block / quality_) * quality_
            compressed_block = cv2.idct(dct_block_quantized) * 255.0
            compressed_block = np.clip(compressed_block, 0, 255)
            compressed_img[i:i + block_size_, j:j + block_size_] = np.uint8(compressed_block)

    return compressed_img

if __name__ == "__main__":
    input_image_path = 'tiles.jpg'
    output_image_path = 'compressed_tiles.jpg'
    quality = 1
    block_size = 8
    compressed_image = compress_image(input_image_path, quality, block_size)

    cv2.imwrite(output_image_path, compressed_image)

    print("Image compressed and saved successfully.")

    cv2.imshow('Original Image', cv2.resize(cv2.imread(input_image_path), (800, 600)))
    cv2.imshow('Compressed Image', cv2.resize(compressed_image, (800, 600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
