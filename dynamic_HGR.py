import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import cv2
import mediapipe as mp
import numpy as np
import time

# Instatiate landmark - related 
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Define model x HGR
class GestClassifier(nn.Module):
  def __init__(self):
    super(GestClassifier, self).__init__()
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(21 * 3, 42)
    self.fc2 = nn.Linear(42, 21)
    self.fc3 = nn.Linear(21, 6)
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    x = self.flatten(x)
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.softmax(self.fc3(x))
    return x
  
# Instatiate model and load the saved state dict
model = GestClassifier()
model_state_dict = torch.load('best_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(model_state_dict)
model.eval() # we're not going to use the model in training mode

# Load gesture dictionaries
gestures = ['two_up', 'fist', 'two_down', 'stop', 'dislike', 'like']
gesture_to_num = {gesture: num for num, gesture in enumerate(gestures)} 
num_to_gesture = {num: gesture for gesture, num in gesture_to_num .items()} 
    
def draw_landmarks(image, results):
    for hand_landmarks in results.multi_hand_landmarks:
      mp_drawing.draw_landmarks(
        image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())
    cv2.imshow('Video', image)

def predict(model, landmark_list):
  with torch.no_grad():
    outputs = model(landmark_list)
    # Get the prediction of each frame in the list
    _, predicted = torch.max(outputs, 1)
    # Compute the mode of labels i.e. the most present gesture and return it
    mode, _ = torch.mode(predicted.view(-1))
    print(f'Predicted gesture: {num_to_gesture[mode.item()]}')

# Get reference of main webcam
video_capture = cv2.VideoCapture(0)
landmarks = []
last_prediction_time = time.time()

with mp_hands.Hands(
  max_num_hands = 1,
  model_complexity = 0,
  min_detection_confidence = 0.5,
  min_tracking_confidence = 0.5) as hands:
   
  while video_capture.isOpened():
      ref, frame = video_capture.read()
      # enable 'selfie view'
      frame = cv2.flip(frame, 1)
      if not ref:
        print("Ignoring empty camera frame.")
        break
      
      frame.flags.writeable = False
      results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
      frame.flags.writeable = True
     
      # If a right hand was detected, get landmarks and append to list
      if results.multi_hand_landmarks and results.multi_handedness[0].classification[0].label== 'Right':
        #print(f"{results.multi_handedness[0].classification[0].label}")# == 'Right':
        current_time = time.time()
        if current_time - last_prediction_time < 2:
          cv2.imshow('Video', frame)
        else:
          for landmark in results.multi_hand_landmarks[0].landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
          
          draw_landmarks(frame, results)
          # When 30 frames have hands, recognize gesture
          
          if len(landmarks) == 30*21: 
            # Transform landmarks in torch tensor of the correct shape
            landmarks = torch.tensor(landmarks).view(30,21,3)
            predict(model, landmarks)
            # Empty landmarks list
            landmarks = []
            last_prediction_time = time.time()
            #time.sleep(2)
      else:
        cv2.imshow('Video', frame)

      if cv2.waitKey(5) & 0xFF == 27:
        break
video_capture.release()