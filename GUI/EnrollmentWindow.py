import os
import io
import pickle
import face_recognition
from PIL import Image

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
        self.setWindowIcon(QIcon("logos_graphics/gestotune_logo.png"))
    
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
        self.name = self.name.lower()
    
    def saveSurname(self) -> None:
        # function stores user's surname
        self.surname = self.textEdit_2.toPlainText()
        self.surname = self.surname.lower()

    def savePlaylistID(self) -> None:
        # function stores playlist ID chosen by the user
        self.playlist_id = self.textEdit_3.toPlainText()

    def save_encodings(self, encodings: list, names: list) -> None:
        # function saves known face encodings and names to a file
        with open("enrollment_files/face_encodings.pkl", "wb") as f:
            pickle.dump({"encodings": encodings, "names": names}, f)

    def load_encodings(self):
        # function loads existing face encodings and names from a file
        if os.path.exists("enrollment_files/face_encodings.pkl"):
            with open("enrollment_files/face_encodings.pkl", "rb") as f:
                data = pickle.load(f)
                return data["encodings"], data["names"]
        return [], []

    def take_photo(self) -> None:
        # function takes user's photo for enrollment and computes and stores face embeddings of new user
        _, frame = self.takeFace.cap.read()

        """ Store photo taken on device
        photo = self.takeFace.convert_image(frame)
        filename = "enrollment_files/"+self.name+"_"+self.surname+".png"
        photo.save(filename, "png")
        """
        pil_image = Image.fromarray(frame)
        with io.BytesIO() as byte_io:   # convert PIL Image to bytes (avoids saving image)
            pil_image.save(byte_io, format='JPEG')
            byte_io.seek(0)
            image_bytes = byte_io.read()

        # compute face embeddings
        user_image = face_recognition.load_image_file(io.BytesIO(image_bytes))
        user_face_encoding = face_recognition.face_encodings(user_image)[0]

        # load existing encodings + add new user and save
        known_face_encodings, known_face_names = self.load_encodings()
        known_face_encodings.append(user_face_encoding)
        known_face_names.append(f"{self.name}_{self.surname}")
        self.save_encodings(known_face_encodings, known_face_names)

        self.pushButton_2.setText("PHOTO TAKEN")
        self.pushButton_2.setEnabled(False)

    def load_preferences(self):
        # function loads existing playlist id chosen by previous users
        if os.path.exists("enrollment_files/preferences.pkl"):
            with open("enrollment_files/preferences.pkl", "rb") as f:
                data = pickle.load(f)
                return data["names"], data["playlist_ids"]
        return [], []

    def save_preferences(self, names: list, playlist_id: list) -> None:
        # function saves in a file the playlist id chosen by each user
        with open("enrollment_files/preferences.pkl", "wb") as f:
            pickle.dump({"names": names, "playlist_ids": playlist_id}, f)

    def enroll_method(self) -> None:
        # function moves forward to the face detection phase
        if self.name and self.surname and self.playlist_id:
            self.faceDetection_window = FaceDetectionWindow()
            known_names, known_playlistIDS = self.load_preferences()
            
            if f"{self.name}_{self.surname}" not in known_names:
                known_names.append(f"{self.name}_{self.surname}")
                known_playlistIDS.append(self.playlist_id)
            else:
                known_playlistIDS[known_names.index(f"{self.name}_{self.surname}")] = self.playlist_id
            
            self.save_preferences(known_names, known_playlistIDS)
            
            self.faceDetection_window.token = self.token
            self.close()
            self.faceDetection_window.show()
            