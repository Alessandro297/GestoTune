from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QWidget

from GUI.EnrollmentWindow import EnrollmentWindow
from GUI.FaceDetectionWindow import FaceDetectionWindow
from PyUI.StartWindow import Ui_MainWindow
from components.Spotify import SpotifyAPI

class MainWindow(QMainWindow, Ui_MainWindow):
    # This class represents main window GUI

    def __init__(self) -> None:
        """
        MainWindow constructor
        """
        super().__init__()
        self.setupUi(self)
        self.token = ""
        self.spotify_connection = SpotifyAPI()
        self.enrollment_window = None
        self.faceDetection_window = None
        self._connect_buttons()
        self.setWindowIcon(QIcon("logos_graphics/gestotune_logo.png"))
    
    def _connect_buttons(self) -> None:
        # function connects buttons with their actions
        self.start_btn.clicked.connect(self._on_start_button_clicked)
        self.login_btn.clicked.connect(self._on_login_button_clicked)
        self.pushButton.clicked.connect(self._on_enroll_button_clicked)
    
    def _on_start_button_clicked(self) -> None:
        # function opens face detection window
        if self.token:
            self.faceDetection_window = FaceDetectionWindow()
            self.faceDetection_window.token = self.token
            self.faceDetection_window.show()
        else:
            self.label.setText("Please login via Spotify account")

    def _on_login_button_clicked(self) -> None:
        # function checks whether user is logged in
        # returns true if user logged in, false otherwise
        self.spotify_connection.change_token.connect(self.update_token)
        self.spotify_connection.change_msg.connect(self.update_msg)
        self.spotify_connection.start()

    def _on_enroll_button_clicked(self) -> None:
        # function opens enrollment window
        if self.token:
            self.enrollment_window = EnrollmentWindow()
            self.enrollment_window.token = self.token
            self.enrollment_window.show()
        else:
            self.label.setText("Please login via Spotify account")
    
    @pyqtSlot(str)
    def update_token(self, token):
        self.token = token
    
    @pyqtSlot(str)
    def update_msg(self, msg):
        self.label.setText(msg)
