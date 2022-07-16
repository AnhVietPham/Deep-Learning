import cv2 as cv
import numpy as np

if __name__ == '__main__':
    img = cv.imread('images/cat.jpg')
    cv.imshow('Cat', img)

    b, g, r = cv.split(img)
    cv.imshow('Blue', b)
    cv.imshow('Green', g)
    cv.imshow('Red', r)

    merged = cv.merge([b, g, r])
    cv.imshow('Merge Image', merged)

    merged1 = cv.merge([g, r, b])
    cv.imshow('Merge Image', merged1)

    blank = np.zeros(img.shape[:2], dtype='uint8')
    blue = cv.merge([b, blank, blank])
    green = cv.merge([blank, g, blank])
    red = cv.merge([blank, blank, r])

    cv.imshow('Blue', blue)
    cv.imshow('Green', green)
    cv.imshow('Red', red)

    cv.waitKey()
