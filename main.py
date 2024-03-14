import cv2
import numpy as np
import face_recognition as fr

load_img = fr.load_image_file("dataset/abdul qadeer.jpg")
rgb_img = cv2.cvtColor(load_img, cv2.COLOR_BGR2RGB)

face_locations = fr.face_locations(rgb_img)[0]
copy = rgb_img.copy()

cv2.rectangle(copy, (face_locations[3], face_locations[0]),(face_locations[1], face_locations[2]), (255,0,255), 2)

cv2.imshow('copy', copy)
cv2.imshow('img', rgb_img)

cv2.waitKey(0)
