import cv2
from random import randrange

# load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# choose an image to detect faces in
img = cv2.imread('Chris_Hemsworth.jpg')
# img = cv2.imread('chris.jpg')

# converting to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# draw rectangles around a single face
# (x,y,w,h) = face_coordinates[0]
# cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)

# draw rectangle around multiple faces
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w, y+h), (randrange(256),randrange(256),randrange(256)), 2)

# print(face_coordinates)
cv2.imshow('Chris Hemsworth', img)
cv2.waitKey()

print("Code Completed")