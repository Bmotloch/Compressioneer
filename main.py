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
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

