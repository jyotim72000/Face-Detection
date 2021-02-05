import cv2
from random import randrange

# load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
"""from video"""
# # to capture faces from a video
# webcam = cv2.VideoCapture('video.mp4')

# to capture video from webcam
webcam = cv2.VideoCapture(0)

# iterate forever over frames
while True:
    #read the current frame
    successful_frame_read, frame = webcam.read()
    # converting to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # draw rectangle around multiple faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(128, 256), randrange(128, 256), randrange(128, 256)), 2)

    cv2.imshow('Chris Hemsworth', frame)
    key = cv2.waitKey(1)

    # stop if Q key is pressed
    if key==80 or key==113:
        break

# release the VideoCapture object
webcam.release()

"""form image"""
'''
# choose an image to detect faces in
# img = cv2.imread('Chris_Hemsworth.jpg')
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
    cv2.rectangle(img, (x,y), (x+w, y+h), (randrange(128,256),randrange(128,256),randrange(128,256)), 2)

# print(face_coordinates)
# display the image
cv2.imshow('Chris Hemsworth', img)
cv2.waitKey()

print("Code Completed")
'''