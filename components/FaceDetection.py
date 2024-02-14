import sys
import os
import cv2
import numpy as np
import pickle
import face_recognition
import time

from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from components.Spotify import SpotifyAPI

class FaceDetectionThread(QThread):
    # This class represents the camera thread for face detection window

    def __init__(self) -> None:
        """
        FaceDetectionThread constructor
        """
        super(FaceDetectionThread, self).__init__()
        self.cap = cv2.VideoCapture(0)
        self.is_running = True
        self.token = ""
        self.preferences = {}   # dict keeps track of the playlist id chosen by each user in the enrollment phase (name_surname : playlist_id)
    
    change_face_name = pyqtSignal(str)
    change_playlist = pyqtSignal(str)
    change_image = pyqtSignal(QImage)
    clear_labels = pyqtSignal()

    def convert_image(self, image: np.ndarray) -> QImage:
        # function converts np.ndarray fromm camera to QImage to display it
        image = cv2.flip(image, 1)
        RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = RGB_image.shape
        converted_image = QImage(RGB_image.data, w, h, QImage.Format_RGB888)
        scaled_image = converted_image.scaled(600, 600, Qt.KeepAspectRatio)
        return scaled_image
    
    def loadPreferences(self) -> None:
        # function loads in a dict the playlist id chosen by each user in the enrollment phase
        if os.path.exists("enrollment_files/preferences.pkl"):
            with open("enrollment_files/preferences.pkl", "rb") as f:
                data = pickle.load(f)
                self.preferences = dict(zip(data["names"], data["playlist_ids"]))
        # print(self.preferences)
    
    def load_encodings(self):
        # function loads existing face encodings and names from a file
        if os.path.exists("enrollment_files/face_encodings.pkl"):
            with open("enrollment_files/face_encodings.pkl", "rb") as f:
                data = pickle.load(f)
                return data["encodings"], data["names"]
        return [], []

    def run(self) -> None:
        # function detect faces based on image from camera and add detected face's name and playlist's name to the queue
        spotify = SpotifyAPI(token=self.token)

        self.loadPreferences()
        known_face_encodings, known_face_names = self.load_encodings()
        face_locations = []
        face_encodings = []
        last_predictionTime = time.time()

        while self.cap.isOpened() and self.is_running:
            ret, frame = self.cap.read()

            if not ret:
                print("Problem with camera...exiting...")
                break
            
            current_time = time.time()

            if current_time - last_predictionTime > 3:
                # face recognition
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(np.array(rgb_small_frame), face_locations)
                for face_encoding in face_encodings:
                    # check if the face is a match for the known ones
                    # here we could modify tollerance level
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                    playlist = "Unknown"
                    # pick the known face with smallest (Euclidean) distance to the new one
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_idx = np.argmin(face_distances)
                    if matches[best_idx]:
                        # find the name of the person
                        name = known_face_names[best_idx]
                        playlist = spotify.get_playlist_name(self.preferences[name])
                        
                    self.change_face_name.emit(name)
                    self.change_playlist.emit(playlist)
                last_predictionTime = time.time()
                    
            self.change_image.emit(self.convert_image(frame))
            
            cv2.waitKey(10)
        
        self.cap.release()
        self.clear_labels.emit()
