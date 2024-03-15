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

cv2.imshow("stored_images", dataset_images[3])
cv2.waitKey(0)




# face_locations = fr.face_locations(rgb_img)[0]
# copy = rgb_img.copy()
#
# cv2.rectangle(copy, (face_locations[3], face_locations[0]),(face_locations[1], face_locations[2]), (255,0,255), 2)
#
# cv2.imshow('copy', copy)
# cv2.imshow('img', rgb_img)
