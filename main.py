import os
import cv2
import numpy as np
import face_recognition as fr

# Load images from the dataset directory
def load_images_from_directory(directory):
    images = []
    image_files = os.listdir(directory)
    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        image = fr.load_image_file(image_path)
        images.append(image)
    return images

# Load known face encodings
def load_known_face_encodings(dataset_images):
    known_face_encodings = []
    for image in dataset_images:
        face_encoding = fr.face_encodings(image)[0]
        known_face_encodings.append(face_encoding)
    return known_face_encodings

# Load images from the dataset
dataset_dir = './dataset'
dataset_images = load_images_from_directory(dataset_dir)

# Generate encodings for the images in the dataset
known_face_encodings = load_known_face_encodings(dataset_images)

# Open the default camera
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    # rgb_frame = frame[:, :, ::-1]

    flipped_frame = cv2.flip(frame, 1)

    face_locations = fr.face_locations(flipped_frame)

    face_encodings = fr.face_encodings(frame, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = fr.compare_faces(known_face_encodings, face_encoding)
        if True in matches:
            match_index = matches.index(True)
            # Known face: Green rectangle
            rectangle_color = (0, 255, 0)
            text_color = (0, 255, 0)
            label = "Known Face"
        else:
            # Unknown face: Red rectangle
            rectangle_color = (0, 0, 255)
            text_color = (0, 0, 255)
            label = "Unknown Face"

        # Draw a rectangle around the face
        cv2.rectangle(flipped_frame, (left, top), (right, bottom), rectangle_color, 2)

        # Draw the text
        cv2.putText(flipped_frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    # Display the resulting frame
    cv2.imshow('Video', flipped_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
