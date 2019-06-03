# OpenCV
import cv2

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Load xml file for face detection
haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def _train(training_set_folder='training'):
    print("Work In Progress")

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


def face_detection(img, scaleFactor=1.1, width=15):
    # Convert to gray scale
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = haar_cascade_face.detectMultiScale(img_grey, scaleFactor=scaleFactor, minNeighbors=5)

    for (x, y, w, h) in faces:
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
    ''' Face Detection on Video '''
    show_video('gaben.mp4')

    ''' Face Detection on Webcam '''
    # show_webcam(mirror=True)

if __name__ == "__main__":
    main()
