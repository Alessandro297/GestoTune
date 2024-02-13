from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QCloseEvent, QIcon, QImage, QKeyEvent, QPixmap
from PyQt5.QtWidgets import QDialog, QWidget, QTextEdit, QFileDialog

from GUI.FaceDetectionWindow import FaceDetectionWindow
from PyUI.EnrollmentWindow import Ui_DetectionWindow
from components.Enrollment import EnrollmentThread

class EnrollmentWindow(QDialog, Ui_DetectionWindow):
    # This class represents enrollment window GUI

    def __init__(self) -> None:
        """
        EnrollmentWindow constructor
        """
        super().__init__()
        self.setupUi(self)
        self.token = ""
        self.name = ""
        self.surname = ""
        self.playlist_id = ""
        self.faceDetection_window = None
        self.takeFace = EnrollmentThread()
        self._store_text()
        self._connect_buttons()
        self._connect_camera()
        self.setWindowIcon(QIcon("GUI/face.png"))
    
    def _connect_buttons(self) -> None:
        # function connects buttons with their actions
        self.pushButton_2.clicked.connect(self.take_photo)
        self.pushButton_3.clicked.connect(self.enroll_method)

    def _connect_camera(self) -> None:
        # function starts camera
        self.takeFace.change_image.connect(self.update_image)
        self.takeFace.clear_labels.connect(self.clear_labels)
        self.takeFace.start()
    
    def closeEvent(self, event) -> None:
        # function overrides method in QDialog: finish detection QThread and close EnrollmentWindow
        self.takeFace.is_running = False
        self.takeFace = EnrollmentThread()
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
        # function clear labels after close Enrollment window
        self.detection_camera.clear()
        self.detection_camera.setText("TAKE PHOTO TO REGISTER YOUR FACE")
        self.detection_camera.setAlignment(Qt.AlignCenter)

    def _store_text(self) -> None:
        # function stores informations filled in by the user
        self.textEdit.textChanged.connect(self.saveName)
        self.textEdit_2.textChanged.connect(self.saveSurname)
        self.textEdit_3.textChanged.connect(self.savePlaylistID)
    
    def saveName(self) -> None:
        # function stores user's name
        self.name = self.textEdit.toPlainText()
    
    def saveSurname(self) -> None:
        # function stores user's surname
        self.surname = self.textEdit_2.toPlainText()

    def savePlaylistID(self) -> None:
        # function stores playlist ID chosen by the user
        self.playlist_id = self.textEdit_3.toPlainText()

    def take_photo(self) -> None:
        # function takes user's photo for enrollment
        _, frame = self.takeFace.cap.read()
        photo = self.takeFace.convert_image(frame)
        filename = "enrollment_photo/"+self.name+"_"+self.surname+".png"
        photo.save(filename, "png")
        self.pushButton_2.setText("PHOTO TAKEN")

    def enroll_method(self) -> None:
        # function moves forward to the face detection phase
        if self.name and self.surname and self.playlist_id:
            
            # TODO: funzione che manda al modello di face detection la foto scattata (enrollment) con nome e cognome (label) associato
            
            self.faceDetection_window = FaceDetectionWindow()
            self.faceDetection_window.name = self.name
            self.faceDetection_window.surname = self.surname
            self.faceDetection_window.playlist_id = self.playlist_id
            self.faceDetection_window.token = self.token
            self.close()
            self.faceDetection_window.show()
            
    
