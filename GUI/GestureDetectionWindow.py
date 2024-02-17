from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QWidget

from PyUI.GestureDetectionWindow import Ui_DetectionWindow
from components.GestureDetection import GestureDetectionThread

class GestureDetectionWindow(QDialog, Ui_DetectionWindow):
    # This class represents gesture detection window GUI

    def __init__(self) -> None:
        """
        GestureDetectionWindow constructor
        """
        super().__init__()
        self.setupUi(self)
        self.token = ""
        self.playlist_id = ""
        self.gestureDetection = GestureDetectionThread()
        self._connect_buttons()
        self.setWindowIcon(QIcon("GUI/hand.png"))

    def _connect_buttons(self) -> None:
        # function connects buttons with their actions
        self.pushButton_2.clicked.connect(self.connection_action)
    
    def connection_action(self) -> None:
        # function starts actions of created thread
        self.gestureDetection.change_image.connect(self.update_image)
        self.gestureDetection.change_confidence.connect(self.update_confidence)
        self.gestureDetection.change_gesture_name.connect(self.update_gesture_name)
        self.gestureDetection.clear_labels.connect(self.clear_labels)
        self.gestureDetection.token = self.token
        self.gestureDetection.playlist_id = self.playlist_id
        self.gestureDetection.start()

    def closeEvent(self, event) -> None:
        # function overrides method in QDialog: finish detection QThread and close face detection window
        self.gestureDetection.is_running = False
        self.gestureDetection = GestureDetectionThread()
        self.close()
    
    def keyPressEvent(self, event) -> None:
        # function overrides method in QDialog: close after press escape button
        if event.key() == Qt.Key_Escape:
            self.close()

    @pyqtSlot(QImage)
    def update_image(self, image: QImage) -> None:
        # function shows image from camera
        # image: converted frame from camera
        self.detection_camera.setPixmap(QPixmap.fromImage(image))
    
    @pyqtSlot()
    def clear_labels(self) -> None:
        # function clear labels after close gesture detection window
        self.detection_camera.clear()
        self.detection_camera.setText("CLICK START, OPEN SPOTIFY AND START PLAY MUSIC")
        self.detection_camera.setAlignment(Qt.AlignCenter)

    @pyqtSlot(str)
    def update_confidence(self, confidence: str) -> None:
        # function updates confidence of predicted gesture
        self.detection_confidence.setText(confidence)
        self.detection_confidence.setAlignment(Qt.AlignCenter)
    
    @pyqtSlot(str)
    def update_gesture_name(self, gesture_name: str) -> None:
        # function updates detected action
        self.detected_gesture_name.setText(gesture_name)
        self.detected_gesture_name.setAlignment(Qt.AlignCenter)
