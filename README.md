# Compressioneer: Image Compression Tool with DCT & Huffman Coding
## Overview

Compressioneer is a Python desktop application for image compression using Discrete Cosine Transform (DCT) and Huffman coding, implementing a JPEG-like pipeline.
It features a PyQt5 GUI with real-time visualization and adjustable compression parameters.

## Features

- Compression Pipeline: DCT → Quantization → Zigzag → Run-Length Encoding → Huffman coding

- Custom Format: .isa proprietary compressed image format

- Quality Control: Adjustable quality factor (1–100) and block size (1–64)

- Visual Comparison: Side-by-side original vs. compressed images with difference visualization

- Metrics: MSE, PSNR, and file size savings calculations

- Multi-format Support: PGM, ISA, PNG, JPG input/output

## Technical Stack

- Language: Python 3.12

- GUI: PyQt5 with multithreading for responsive UI

### Algorithms:

- Discrete Cosine Transform (DCT)

- Quantization with customizable tables

- Zigzag scanning

- Run-Length Encoding (RLE)

- Huffman coding

- Image Processing: OpenCV, NumPy, SciPy

## Implementation Details

- Block Processing: Divide images into customizable blocks (1×1 to 64×64)

- DCT Transformation: Convert spatial domain to frequency domain

- Quantization: Reduce high-frequency components based on quality factor

- Entropy Coding: RLE + Huffman coding for efficient compression

### Advanced Features:

- Adaptive encoding between direct and RLE

- Custom quantization tables with interpolation

- Real-time loss analysis (MSE & PSNR)

- Multithreaded processing for a non-blocking UI

## Skills Demonstrated

- Algorithm design: DCT, quantization, RLE, Huffman coding

- Digital signal processing and matrix operations

- Data structures: Huffman trees and priority queues

- Software architecture: Modular design separating UI and compression logic

- Performance optimization: Block-wise processing and memory-efficient operations

- File format design: Custom binary format with header metadata

- Multithreading with PyQt5 QThread

## Industry Applications

- Digital media storage and transmission

- Computer vision preprocessing

- Data compression and information theory

- Multimedia systems and custom codec development
