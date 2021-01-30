import cv2 as cv

# load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# choose an image to detect faces in
img = cv.imread('chris.jfif')

# converting to grayscale
grayscaled_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.imshow('Chris Hemsworth', grayscaled_img)
cv.waitKey()

print("Code Completed")