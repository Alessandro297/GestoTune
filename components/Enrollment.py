import sys
import cv2
import numpy as np

from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

class EnrollmentThread(QThread):
    # This class represents the camera thread for enrollment window

    def __init__(self) -> None:
        """
        EnrollmentThread constructor
        """
        super(EnrollmentThread, self).__init__()
        self.cap = cv2.VideoCapture(0)
        self.is_running = True
    
    change_image = pyqtSignal(QImage)
    clear_labels = pyqtSignal()

    def convert_image(self, image: np.ndarray) -> QImage:
        # function converts np.ndarray from camera to QImage to display it
        image = cv2.flip(image, 1)
        RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = RGB_image.shape
        converted_image = QImage(RGB_image.data, w, h, QImage.Format_RGB888)
        scaled_image = converted_image.scaled(600, 600, Qt.KeepAspectRatio)
        return scaled_image

    def run(self) -> None:
        # function detect face based on image from camera
        while self.cap.isOpened() and self.is_running:
            ret, frame = self.cap.read()

            if not ret:
                print("Problem with camera... exiting...")
                break
            
            self.change_image.emit(self.convert_image(frame))

            cv2.waitKey(10)
        
        self.cap.release()
        self.clear_labels.emit()
