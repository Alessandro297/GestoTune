import face_recognition
import cv2
import numpy as np
import argparse
import os
import pickle
import io
from PIL import Image

# function to save known face encodings and names to a file
def save_encodings(encodings, names):
    with open("face_encodings.pkl", "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)

# function to load known face encodings and names from a file
def load_encodings():
    if os.path.exists("face_encodings.pkl"):
        with open("face_encodings.pkl", "rb") as f:
            data = pickle.load(f)
            return data["encodings"], data["names"]
    return [], []

# Enrollment function: new users
def enroll_user():
    # get reference to the default webcam
    video_capture = cv2.VideoCapture(0) 
    print("Press 's' to take the photo. It will be saved in the system for future use")
    while True:
        ret, frame = video_capture.read() # get a frame
        cv2.imshow('Video', frame) 
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

    pil_image = Image.fromarray(frame)
    # convert the PIL Image to bytes (avoids saving image)
    with io.BytesIO() as byte_io:
      pil_image.save(byte_io, format='JPEG')
      byte_io.seek(0)
      image_bytes = byte_io.read()

    user_image = face_recognition.load_image_file(io.BytesIO(image_bytes))
    user_name = input("Enter your name and surname: ")
    user_face_encoding = face_recognition.face_encodings(user_image)[0]

    # load existing encodings + add new user and save
    known_face_encodings, known_face_names = load_encodings()
    known_face_encodings.append(user_face_encoding)
    known_face_names.append(user_name)
    save_encodings(known_face_encodings, known_face_names)

    print(f"Enrolled {user_name} successfully.")

# Login function: saved users
def login_user():
    # load known users + get reference to default webcam
    known_face_encodings, known_face_names = load_encodings()
    video_capture = cv2.VideoCapture(0)

    # initialization
    face_locations = []
    face_encodings = []
    process_this_frame = True

    while True:
        ret, frame = video_capture.read() # get a single frame
        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(np.array(rgb_small_frame), face_locations)
            print(f"{face_encodings}")

            for face_encoding in face_encodings:
                # check if the face is a match for the known ones
                # here we could modify tollerance level
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # pick the known face with smallest (Euclidean) distance to the new one
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                print(f"Face distances obtained: {face_distances}")
                best_idx = np.argmin(face_distances)
                if matches[best_idx]:
                    # find the name of the person
                    name = known_face_names[best_idx]
                    print(f"Welcome {name}!")
                    # close camera 
                    video_capture.release() 
                    cv2.destroyAllWindows()
                    return

        process_this_frame = not process_this_frame
        cv2.imshow('Video', frame)
        # quit pressing 'q' if the camera in case it is still open (shouldn't be necessary)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Main function to parse arguments and call enroll or login
def main():
    parser = argparse.ArgumentParser(description='Face recognition system')
    parser.add_argument('mode', type=str, choices=['enroll', 'login'], help='Mode: enroll or login')
    args = parser.parse_args()

    if args.mode == 'enroll':
        enroll_user()
    elif args.mode == 'login':
        login_user()

if __name__ == '__main__':
    main()