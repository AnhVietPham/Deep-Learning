import os
import cv2 as cv
import numpy as np

people = ['Courtois', 'Cristiano Ronaldo', 'Dybala', 'Kross', 'Lionel Messi', 'Neymar', 'Pogba']
DIR = r'/Users/anhvietpham/Documents/Dev-Chicken/Deep-Learning/Computer-Vision/opencv-course/datasetpeople/train'

haar_cascade = cv.CascadeClassifier("../haar_face.xml")
features = []
labels = []


def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y: y + h, x: x + h]
                features.append(faces_roi)
                labels.append(label)


if __name__ == '__main__':
    create_train()
    print(f'Length of the features: {len(features)}')
    print(f'Length of the labels: {len(labels)}')
