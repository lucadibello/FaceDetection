import cv2

# Load xml file for face recognition
haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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

def face_detection(img, scaleFactor= 1.1, width = 15):
    # Convert to gray scale
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = haar_cascade_face.detectMultiScale(img_grey, scaleFactor = scaleFactor, minNeighbors = 5);    

    for (x, y, w, h) in faces:
        # Draw rects
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), width)

    # Return image with rects
    return img

def main():
    # Start webcam
    show_webcam(mirror=True)

if __name__ == "__main__":
    main()
