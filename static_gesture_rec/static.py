import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import cv2
import mediapipe as mp
import numpy as np
import sys

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Classifier definition
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

# Remember gestures <-> number relation
gestures = ['two_up', 'fist', 'two_down', 'stop', 'dislike', 'like']
gesture_to_num = {gesture: num for num, gesture in enumerate(gestures)} 
num_to_gesture = {num: gesture for gesture, num in gesture_to_num .items()} 


# get reference to the default webcam
video_capture = cv2.VideoCapture(0) 
print("Press 's' to take a photo of a gesture")
while True:
    ret, frame = video_capture.read() # get a frame
    frame = cv2.flip(frame, 1)
    cv2.imshow('Video', frame) # show image in selfie-like view
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break
video_capture.release()
cv2.destroyAllWindows()

cv2.imwrite('prova.jpg', frame)

IMAGE_FILES = ['prova.jpg']
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    #image = cv2.flip(cv2.imread(file), 1)
    image = cv2.imread(file)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    landmarks = []
    if not results.multi_hand_landmarks:
      print(f"No hand detected.")
      # Close everything. We're done
      sys.exit()

    if results.multi_hand_landmarks: # go ahead only if an hand is detected
    # extract landmarks
        for landmark in results.multi_hand_landmarks[0].landmark:
          landmarks.append([landmark.x, landmark.y, landmark.z])

landmarks = np.array(landmarks)
print(f"{landmarks.shape}")
landmarks = torch.Tensor(landmarks).unsqueeze(0)
print(f"{landmarks.shape}")

with torch.no_grad():
  outputs = model(landmarks)
  print(f"{outputs.shape}, {outputs}")
  _, predicted = torch.max(outputs, 1)

  if outputs[0][predicted.item()] < 0.95: #HERE THRESHOLD IS TO BE DEFINED...
    print(f"Unknown gesture.")
  else:
    print(f'Predicted gesture: {num_to_gesture[predicted.item()]}')

print(f"{outputs}")