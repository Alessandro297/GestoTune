import cv2
import numpy as np
import mediapipe as mp
import time
import torch
import torch.nn as nn
import torch.optim as optim

from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage

from constants import ACTIONS, THRESHOLD
from components.Spotify import SpotifyAPI

class GestureClassifier(nn.Module):
    # This class represents the gesture classifier model
    
    def __init__(self) -> None:
        """
        GestureClassifier constructor
        """
        super(GestureClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(21 * 3, 42)
        self.fc2 = nn.Linear(42, 21)
        self.fc3 = nn.Linear(21, 6)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

class GestureDetectionThread(QThread):
    # This class represents gesture recognition based on image from camera

    def __init__(self) -> None:
        """
        GestureDetectionThread constructor
        """
        super(GestureDetectionThread, self).__init__()
        self.token = ""
        self.playlist_id = ""
        self.is_running = True
        self.cap = cv2.VideoCapture(0)
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        
        # load gesture (action) dictionary
        self.num_to_gesture = {num: gesture for num, gesture in enumerate(ACTIONS)}

        # instantiate model and load the saved stated dict
        self.model = GestureClassifier()
        model_state_dict = torch.load("model/best_model.pth", map_location=torch.device("cpu"))
        self.model.load_state_dict(model_state_dict)
        self.model.eval()
 
    change_gesture_name = pyqtSignal(str)
    change_confidence = pyqtSignal(str)
    change_image = pyqtSignal(QImage)
    clear_labels = pyqtSignal()

    def convert_image(self, image: np.ndarray) -> QImage:
        # function converts np.ndarray from camera to QImage to display it
        RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = RGB_image.shape
        converted_image = QImage(RGB_image.data, w, h, QImage.Format_RGB888)
        scaled_image = converted_image.scaled(600, 600, Qt.KeepAspectRatio)
        return scaled_image

    def draw_landmarks(self, image: np.ndarray, results) -> None:
        # function draws landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style())
        self.change_image.emit(self.convert_image(image))
    
    def predict(self, model, landmarks: list):
        # function predicts hand gesture
        with torch.no_grad():
            outputs = model(landmarks)
            # get the prediction of each frame in the list
            confidence, predicted = torch.max(outputs, 1)
            # compute the mode of labels, i.e. the most present gesture, and the confidence
            mode, _ = torch.mode(predicted.view(-1))
            gesture = self.num_to_gesture[mode.item()]
            mean_confidence = torch.mean(confidence[predicted==mode.item()].view(-1))
            return gesture, mean_confidence.item()

    def run(self) -> None:
        # function detects gestures based on image from camera and add detected gesture's name to the queue
        spotify = SpotifyAPI(token=self.token)
        
        landmarks = []
        last_predictionTime = time.time()

        with self.mp_hands.Hands(
            max_num_hands = 1,
            model_complexity = 0,
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5) as hands:

            while self.cap.isOpened() and self.is_running:
                ret, frame = self.cap.read()
                frame = cv2.flip(frame, 1)
                
                if not ret:
                    print("Problem with camera...exiting...")
                    break
                
                frame.flags.writeable = False
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame.flags.writeable = True

                self.change_image.emit(self.convert_image(frame))

                # If a right hand was detected, get landmarks and append to list
                if results.multi_hand_landmarks and results.multi_handedness[0].classification[0].label == "Right":
                    current_time = time.time()
                    if current_time - last_predictionTime >= 2:
                        for landmark in results.multi_hand_landmarks[0].landmark:
                            landmarks.append([landmark.x, landmark.y, landmark.z])
                        
                        self.draw_landmarks(frame, results)

                        # when 30 frames have hands, recognize gesture
                        if len(landmarks) == 30*21:
                            # transform landmarks in torch tensor of the correct shape
                            landmarks = torch.tensor(landmarks).view(30,21,3)
                            gesture, confidence = self.predict(self.model, landmarks)

                            if confidence > THRESHOLD:
                                self.change_gesture_name.emit(gesture)
                                spotify.gesture_action(gesture, self.playlist_id)
                                self.change_confidence.emit(f"{round(confidence*100, 2)}%")
                            else:
                                self.change_gesture_name.emit("Null")
                                self.change_confidence.emit("-")
                            
                            # empty landmarks list
                            landmarks = []
                            last_predictionTime = time.time()

                cv2.waitKey(10)
            
            self.cap.release()
            self.clear_labels.emit()
