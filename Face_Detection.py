import cv2

# load some pre-trained data on face frontals from opencv (haar cascade algorithm)
# trained_face_data = cv2.CascadeClassifier('D:\\PycharmProjects\\firstProg\\face detection\\haarcascade_frontface_default.xml')
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# choose an image to detect faces in
img = cv2.imread('chris.jpg')

# converting to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# draw rectangles around the faces
cv2.rectangle(img, (271,100), (452, 281), (0,255,0), 2)

print(face_coordinates)
cv2.imshow('Chris Hemsworth', img)
cv2.waitKey()

print("Code Completed")