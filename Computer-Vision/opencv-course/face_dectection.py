import cv2 as cv
import numpy as np

if __name__ == "__main__":
    img = cv.imread('images/grouppeople.jpeg')
    cv.imshow('Face Detection', img)

    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow('Gray Face Detection', gray)

    haar_cascade = cv.CascadeClassifier("haar_face.xml")
    face_rect = haar_cascade.detectMultiScale(img, scaleFactor=1.6, minNeighbors=3)
    print(f'Number of face found: {len(face_rect)}')

    for (x, y, w, h) in face_rect:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)

    cv.imshow('Bounding Box Face Detection', img)

    cv.waitKey(0)
