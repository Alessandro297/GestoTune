from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QWidget

from PyUI.GestureDetectionWindow import Ui_DetectionWindow


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

        # self.gestureDetection = GestureDetectionThread()
        self._connect_buttons()
        self.setWindowIcon(QIcon("GUI/hand.png"))

    def _connect_buttons(self) -> None:
        # function connects buttons with their actions
        self.pushButton_2.clicked.connect(self.connection_action)
    
    def connection_action(self) -> None:
        # function starts actions of created thread
        
        # TODO
        pass

