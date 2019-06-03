import cv2

haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    
    while True:
        ret_val, img = cam.read()

        # Show image
        img = face_detection(img, width = 10)

        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('Hello there', img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit

    cv2.destroyAllWindows()

def face_detection(img, scaleFactor= 1.1, width = 15):
    # Convert to gray scale
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces_rects = haar_cascade_face.detectMultiScale(img_grey, scaleFactor = scaleFactor, minNeighbors = 5);    

    for (x, y, w, h) in faces_rects:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), width)

    return img

def main():
    show_webcam(mirror=True)

if __name__ == "__main__":
    main()
