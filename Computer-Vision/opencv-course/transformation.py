import cv2 as cv
import numpy as np


# -x ==> Left
# -y ==> Up
# x ==> Right
# y ==> Down

def translate(img, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)


def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]
    if rotPoint is None:
        rotPoint = (width // 2, height // 2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimension = (width, height)

    return cv.warpAffine(img, rotMat, dimension)


if __name__ == '__main__':
    img = cv.imread('images/cat.jpg')
    cv.imshow('Cat', img)

    translate = translate(img, 200, 200)
    cv.imshow('Translate', translate)

    rotated = rotate(img, 45)
    cv.imshow('Rotated', rotated)

    cropped = img[200:400, 300:400]
    cv.imshow('Cropped', cropped)

    cv.waitKey(0)
