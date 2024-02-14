import os

from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QWidget

from GUI.GestureDetectionWindow import GestureDetectionWindow
from PyUI.FaceDetectionWindow import Ui_DetectionWindow
from components.FaceDetection import FaceDetectionThread

class FaceDetectionWindow(QDialog, Ui_DetectionWindow):
    # This class represents face detection window GUI

    def __init__(self) -> None:
        """
        FaceDetectionWindow constructor
        """
        super().__init__()
        self.setupUi(self)
        self.token = ""
        self.playlist_id = ""
        self.faceDetection = FaceDetectionThread()
        self.gestureDetection_window = None
        self._connect_buttons()
        self.setWindowIcon(QIcon("GUI/face.png"))

    def _connect_buttons(self) -> None:
        # function connects buttons with their actions
        self.pushButton_2.clicked.connect(self.face_detection)
        self.pushButton_3.clicked.connect(self._on_play_button_clicked)

    def face_detection(self) -> None:
        # function starts camera to detect face and load the user's playlist
        if os.path.exists("enrollment_files/preferences.pkl"):
            self.faceDetection.change_image.connect(self.update_image)
            self.faceDetection.change_playlist.connect(self.update_playlist)
            self.faceDetection.change_face_name.connect(self.update_face_name)
            self.faceDetection.clear_labels.connect(self.clear_labels)
            self.faceDetection.token = self.token
            self.faceDetection.start()
            self.pushButton_2.setEnabled(False)
        else:
            self.pushButton_3.setEnabled(False)
            self.detection_camera.setText("USER NOT ENROLLED, PLEASE GO BACK AND ENROL \n\nPRESS ESCAPE BUTTON ON KEYBOARD")
            self.detection_camera.setAlignment(Qt.AlignCenter)

    def closeEvent(self, event) -> None:
        # function overrides method in QDialog: finish detection QThread and close face detection window
        self.faceDetection.is_running = False
        self.faceDetection = FaceDetectionThread()
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

    @pyqtSlot(str)
    def update_playlist(self, playlist_name: str) -> None:
        # function updates playlist name
        self.detection_playlist.setText(playlist_name)
        self.detection_playlist.setAlignment(Qt.AlignCenter)
    
    def check_user_name(self) -> None:
        # function checks if user is unknown
        if self.detected_face_name.text() == "Unknown":
            self.pushButton_3.setEnabled(False)
            self.label_3.setText("GO BACK AND ENROL")
            self.label_3.setStyleSheet("color: red")
            self.label_3.setAlignment(Qt.AlignCenter)
        else:
            self.pushButton_3.setEnabled(True)
            self.label_3.setText("INFORMATION")
            self.label_3.setStyleSheet("color: white")
            self.label_3.setAlignment(Qt.AlignCenter)
    
    @pyqtSlot(str)
    def update_face_name(self, face_name: str) -> None:
        # function updates label with face name
        self.detected_face_name.setText(face_name)
        self.detected_face_name.setAlignment(Qt.AlignCenter)
        self.check_user_name()
        
    @pyqtSlot()
    def clear_labels(self) -> None:
        # function clear labels after close face detection window
        self.detection_camera.clear()
        self.detection_camera.setText("CLICK START TO DETECT YOUR FACE AND LOAD YOUR PLAYLIST")
        self.detection_camera.setAlignment(Qt.AlignCenter)

    def _on_play_button_clicked(self) -> None:
        # function moves forward to the gesture recognition phase
        if not self.pushButton_2.isEnabled():
            self.gestureDetection_window = GestureDetectionWindow()
            self.gestureDetection_window.playlist_id = self.faceDetection.preferences[self.detected_face_name.text()]
            self.gestureDetection_window.token = self.token
            self.close()
            self.gestureDetection_window.show()
