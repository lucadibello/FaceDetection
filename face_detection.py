# OpenCV
import cv2

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Numpy and OS
import numpy as np
import os

# Load xml file for face detection
haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def _show_image(mat):
    while True:
        cv2.imshow("mat", mat)

        # Set exit key (esc to quit)
        if cv2.waitKey(1) == 27:
            break


def _train(training_set_folder='training'):

    ''' Selects only the face of all the images '''
    images = []
    for found_path in os.listdir(training_set_folder):
        img = cv2.imread("training/" + found_path)

        # resize image (300x300)
        img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)

        # analyze image (get all faces location in the photo)
        faces = _get_face_locations(img)
        print("Found {} face(s) in {} image".format(len(faces), found_path))

        # select part of image
        for (x, y, w, h) in faces:
            img = img[y:y+h, x:x+w]
            images.append(img)

    for img in images:
        _show_image(img)

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    
    while True:
        # Read webcam frame
        ret_val, img = cam.read()

        # Show image
        img = face_detection(img, width = 10)

        if mirror:
            # Flip webcam image (mirror)
            img = cv2.flip(img, 1)

        # Show webcam output in a window
        cv2.imshow('Hello there', img)
        
        # Set exit key (esc to quit)
        if cv2.waitKey(1) == 27: 
            break

    cv2.destroyAllWindows()


def _get_face_locations(img, scaleFactor=1.1):
    # Convert to gray scale
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = haar_cascade_face.detectMultiScale(img_grey, scaleFactor=scaleFactor, minNeighbors=5)

    return faces


def face_detection(img, scaleFactor=1.1, width=15):

    for (x, y, w, h) in _get_face_locations(img, scaleFactor):
        # Draw rects
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), width)

    # Return image with rects
    return img


def show_video(video_path):
    cap = cv2.VideoCapture(video_path)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            frame = face_detection(frame, scaleFactor=1.5, width=10)

            # Display the resulting frame
            cv2.imshow(video_path, frame)

            # Set exit key (esc to quit)
            if cv2.waitKey(1) == 27:
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

def main():
    _train()
    ''' Face Detection on Video '''
    #show_video('gaben.mp4')

    ''' Face Detection on Webcam '''
    # show_webcam(mirror=True)

if __name__ == "__main__":
    main()
