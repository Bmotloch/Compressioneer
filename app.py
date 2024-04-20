from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton,
    QMainWindow, QVBoxLayout, QHBoxLayout, QFileDialog, QSlider
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5 import QtGui
import compressor
import os

import helpers


class OpenFile(QThread):
    finished = pyqtSignal()
    opening_error = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.opened_image = None
        self.file_path = file_path

    def run(self):
        try:
            filename = os.path.basename(self.file_path)
            self.opened_image = compressor.open_image(self.file_path)
            self.finished.emit()
        except Exception as e:
            self.opening_error.emit(str(e))


class Compress(QThread):
    compressed = pyqtSignal(object)
    compression_failed = pyqtSignal()

    def __init__(self, file_path, quality, block_size):
        super().__init__()
        self.file_path = file_path
        self.quality = quality
        self.block_size = block_size

    def run(self):
        try:
            compressed_image = compressor.perform_dct(self.file_path, self.quality, self.block_size)
            if compressed_image is not None:
                self.compressed.emit(compressed_image)
            else:
                self.compression_failed.emit()
        except Exception as e:
            print("Compression error:", e)
            self.compression_failed.emit()


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.open_file_thread = None
        self.compress_thread = None
        self.input_image_path = 'baboon.pgm'
        self.output_image_path = 'baboon.isa'
        self.quality = 50
        self.block_size = 8
        self.compressed_quality = 50
        self.compressed_block_size = 8
        self.setWindowIcon(QtGui.QIcon('assets/icon.jpg'))
        self.chosen_image = None
        self.compressed_image = None
        self.max_display_width = 600
        self.temp_imp = QtGui.QPixmap("assets/hourglass.png").scaledToWidth(250)

        self.file_button = QPushButton()
        self.file_button.setText("Open file")
        self.file_button.clicked.connect(self.choose_file)

        self.compress_button = QPushButton()
        self.compress_button.setText("Start compression")
        self.compress_button.setEnabled(False)
        self.compress_button.clicked.connect(self.compress)

        self.save_button = QPushButton()
        self.save_button.setText("Save as")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_file_as)

        self.quality_slider = QSlider(Qt.Horizontal)
        self.quality_slider.setRange(1, 100)
        self.quality_slider.setValue(50)
        self.quality_slider.setTickInterval(5)
        self.quality_slider.valueChanged.connect(self.set_quality)

        self.quality_slider_label = QLabel()
        self.quality_slider_label.setText(f"Quality: {self.quality}")

        self.quality_slider_layout = QVBoxLayout()
        self.quality_slider_layout.addWidget(self.quality_slider_label)
        self.quality_slider_layout.addWidget(self.quality_slider)

        self.block_size_slider = QSlider(Qt.Horizontal)
        self.block_size_slider.setRange(1, 32)
        self.block_size_slider.setValue(8)
        self.block_size_slider.setTickInterval(4)
        self.block_size_slider.valueChanged.connect(self.set_block_size)

        self.block_size_slider_label = QLabel()
        self.block_size_slider_label.setText(f"Block size: {self.block_size}")

        self.block_size_slider_layout = QVBoxLayout()
        self.block_size_slider_layout.addWidget(self.block_size_slider_label)
        self.block_size_slider_layout.addWidget(self.block_size_slider)

        self.chosen_temp_label = QLabel()
        self.chosen_temp_label.setAlignment(Qt.AlignCenter)
        self.chosen_temp_label.setPixmap(self.temp_imp)
        self.chosen_temp_label.setVisible(False)
        self.chosen_image_display = QLabel()
        self.chosen_image_display.setAlignment(Qt.AlignCenter)
        self.chosen_image_label = QLabel("Original Image")
        self.chosen_image_label.setAlignment(Qt.AlignCenter)

        self.chosen_image_layout = QVBoxLayout()
        self.chosen_image_layout.addWidget(self.chosen_temp_label)
        self.chosen_image_layout.addWidget(self.chosen_image_display)
        self.chosen_image_layout.addWidget(self.chosen_image_label)

        self.compressed_temp_label = QLabel()
        self.compressed_temp_label.setAlignment(Qt.AlignCenter)
        self.compressed_temp_label.setPixmap(self.temp_imp)
        self.compressed_temp_label.setVisible(False)
        self.compressed_image_display = QLabel()
        self.compressed_image_display.setAlignment(Qt.AlignCenter)
        self.compressed_image_label = QLabel("Compressed Image")
        self.compressed_image_label.setAlignment(Qt.AlignCenter)

        self.compressed_image_layout = QVBoxLayout()
        self.compressed_image_layout.addWidget(self.compressed_temp_label)
        self.compressed_image_layout.addWidget(self.compressed_image_display)
        self.compressed_image_layout.addWidget(self.compressed_image_label)

        self.image_layout = QHBoxLayout()
        self.image_layout.addLayout(self.chosen_image_layout)
        self.image_layout.addLayout(self.compressed_image_layout)

        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignCenter)
        self.metrics_label = QLabel()
        self.metrics_label.setAlignment(Qt.AlignCenter)
        self.size_label = QLabel()
        self.size_label.setAlignment(Qt.AlignCenter)

        self.button_layout = QVBoxLayout()
        self.button_layout.addWidget(self.file_button)
        self.button_layout.addWidget(self.compress_button)
        self.button_layout.addWidget(self.save_button)
        self.button_layout.addStretch()

        self.sliderLayout = QVBoxLayout()
        self.sliderLayout.addLayout(self.quality_slider_layout)
        self.sliderLayout.addLayout(self.block_size_slider_layout)
        self.sliderLayout.addStretch()

        self.button_slider_layout = QHBoxLayout()
        self.button_slider_layout.addLayout(self.button_layout)
        self.button_slider_layout.addLayout(self.sliderLayout)

        containerLayout = QVBoxLayout()
        containerLayout.addLayout(self.button_slider_layout)
        containerLayout.addLayout(self.image_layout)
        containerLayout.addWidget(self.info_label)
        containerLayout.addWidget(self.metrics_label)
        containerLayout.addWidget(self.size_label)
        mainContainer = QWidget()
        mainContainer.setLayout(containerLayout)
        self.setCentralWidget(mainContainer)

        self.setGeometry(320, 180, 1280, 720)
        self.setWindowTitle("Compressioneer")

    def choose_file(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.pgm *.isa)")
        file_dialog.setViewMode(QFileDialog.Detail)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_():
            self.input_image_path = file_dialog.selectedFiles()[0]
            filename = os.path.basename(self.input_image_path)
            self.print_info(f'Opening image {filename}...')
            self.chosen_image_display.setVisible(False)
            self.chosen_temp_label.setVisible(True)
            self.open_file_thread = OpenFile(self.input_image_path)
            self.open_file_thread.finished.connect(self.open_file_finished)
            self.open_file_thread.opening_error.connect(self.open_file_error)
            self.open_file_thread.start()

    def open_file_finished(self):
        filename = os.path.basename(self.input_image_path)
        self.chosen_image = self.open_file_thread.opened_image
        self.print_info(f'Successfully opened image {filename}')
        self.chosen_temp_label.setVisible(False)
        self.chosen_image_display.setVisible(True)
        self.display_chosen_image()
        self.compress_button.setEnabled(True)

    def open_file_error(self, error_message):
        self.print_info(f'Error opening file: {error_message}')

    def compress(self):
        filename = os.path.basename(self.input_image_path)
        self.print_info('Compression started...')
        self.size_label.hide()
        self.compressed_image_display.setVisible(False)
        self.compressed_temp_label.setVisible(True)
        self.compress_thread = Compress(self.input_image_path, self.quality, self.block_size)
        self.compress_thread.compressed.connect(self.compression_finished)
        self.compress_thread.compression_failed.connect(self.compression_failed)
        self.compress_thread.start()

    def compression_finished(self, compressed_image):
        filename = os.path.basename(self.input_image_path)
        self.compressed_image = compressed_image
        self.compressed_quality = self.quality
        self.compressed_block_size = self.block_size
        self.save_button.setEnabled(True)
        self.print_info(f'Image {filename} compressed successfully!')
        self.compressed_temp_label.setVisible(False)
        self.compressed_image_display.setVisible(True)
        self.display_compressed_image()
        self.print_metrics()

    def compression_failed(self):
        self.print_info('Compression failed. Please try again.')

    def save_file_as(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.pgm *.isa)")
        file_dialog.setViewMode(QFileDialog.Detail)
        file_dialog.setFileMode(QFileDialog.AnyFile)
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        if file_dialog.exec_():
            self.output_image_path = file_dialog.selectedFiles()[0]
            self.metrics_label.hide()
            self.print_info('Saving...')
            compressor.save_image(self.compressed_image, self.output_image_path, self.compressed_quality,
                                  self.compressed_block_size)
            filename = os.path.basename(self.output_image_path)
            self.print_info(f'Image saved as: {filename}')
            self.print_size_dif()
            self.save_button.setEnabled(False)

    def print_info(self, text):
        self.info_label.setText(text)

    def print_metrics(self):
        mse = helpers.calculate_mse(self.chosen_image, self.compressed_image)
        psnr = helpers.calculate_psnr(self.chosen_image, self.compressed_image)
        self.metrics_label.show()
        self.metrics_label.setText(f"MSE between original and compressed images: {round(mse, 3)}\n"
                                   f"PSNR between original and compressed images: {round(psnr, 3)}dB")

    def print_size_dif(self):
        original_size = round(os.path.getsize(self.input_image_path) / 1024)
        compressed_size = round(os.path.getsize(self.output_image_path) / 1024)
        dif = original_size - compressed_size
        text = ""
        if dif > 0:
            text = f"Saved {dif}kb"
        elif dif < 0:
            text = f"Lost {dif}kb"
        else:
            text = "Size unchanged"
        self.size_label.show()
        self.size_label.setText(f"Original size: {original_size}kb\n"
                                f"Compressed size: {compressed_size}kb\n"
                                f"{text}")

    def set_quality(self):
        self.quality = self.quality_slider.value()
        self.set_quality_label()

    def set_block_size(self):
        self.block_size = self.block_size_slider.value()
        self.set_block_size_label()

    def set_quality_label(self):
        self.quality_slider_label.setText(f"Quality: {self.quality}")

    def set_block_size_label(self):
        self.block_size_slider_label.setText(f"Block size: {self.block_size}")

    def display_chosen_image(self):
        height, width = self.chosen_image.shape
        qImg = QtGui.QImage(self.chosen_image.tobytes(), width, height, QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap.fromImage(qImg)

        scale_factor = self.max_display_width / width if width > 0 else 1.0
        pixmap = pixmap.scaledToWidth(int(width * scale_factor))

        self.chosen_image_display.setPixmap(pixmap)

    def display_compressed_image(self):
        height, width = self.compressed_image.shape
        bytes_per_line = width
        qImg = QtGui.QImage(self.compressed_image.tobytes(), width, height, bytes_per_line,
                            QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap.fromImage(qImg)

        scale_factor = self.max_display_width / width if width > 0 else 1.0
        pixmap = pixmap.scaledToWidth(int(width * scale_factor))

        self.compressed_image_display.setPixmap(pixmap)
