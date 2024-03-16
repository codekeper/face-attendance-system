import os
import cv2
import numpy as np
import face_recognition as fr

# load image and change the color scheme from BGR to RGB:
def loop_through_images(directory):
    images = []
    image_files = os.listdir(directory)
    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        image = fr.load_image_file(image_path)
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(rgb_img)
    return images

script_directory = './dataset'
dataset_directory = os.path.abspath(script_directory)
dataset_images = loop_through_images(dataset_directory)

# Open the default camera (usually the first camera connected)
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
    rgb_frame = frame[:, :, ::-1]

    # Find all face locations in the frame
    face_locations = fr.face_locations(rgb_frame)

    # Draw a rectangle around each face
    for top, right, bottom, left in face_locations:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # Flip the frame horizontally
    flipped_frame = cv2.flip(frame, 1)

    cv2.imshow('Flipped Video', flipped_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
