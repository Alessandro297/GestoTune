from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QWidget

from GUI.GestureDetectionWindow import GestureDetectionWindow
from PyUI.FaceDetectionWindow import Ui_DetectionWindow


class FaceDetectionWindow(QDialog, Ui_DetectionWindow):
    # This class represents face detection window GUI

    def __init__(self) -> None:
        """
        FaceDetectionWindow constructor
        """
        super().__init__()
        self.setupUi(self)
        self.token = ""
        self.name = ""
        self.surname = ""
        self.playlist_id = ""
        self.preferences = {}   # dict keeps track of the playlist id chosen by each user in the enrollment phase (name_surname : playlist_id)

        #self.faceDetection = FaceDetectionThread()
        self.gestureDetection_window = None
        self._connect_buttons()
        self.setWindowIcon(QIcon("GUI/face.png"))

    def _connect_buttons(self) -> None:
        # function connects buttons with their actions
        self.pushButton_2.clicked.connect(self.face_detection)
        self.pushButton_3.clicked.connect(self._on_play_button_clicked)
    
    def storePreferences(self) -> None:
        # function stores in a dict the playlist id chosen by each user in the enrollment phase
        if self.name and self.surname and self.playlist_id:
            self.preferences[f"{self.name}_{self.surname}"] = self.playlist_id
        # print(self.preferences)

    def face_detection(self) -> None:
        # function starts camera to detect face and load the user's playlist
        self.storePreferences()
        
        # TODO
        

    def _on_play_button_clicked(self) -> None:
        # function moves forward to the gesture recognition phase

        # TODO
        pass

    
