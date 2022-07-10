import cv2 as cv
import numpy as np

if __name__ == '__main__':
    blank = np.zeros((500, 500, 3), dtype='uint8')
    cv.imshow('Blank', blank)
    # # Draw Height and Width
    # blank[200:300, 200:400] = 0, 255, 0

    # Draw a Rectangle
    # cv.rectangle(blank, (0, 0), (500, 250), (0, 250, 0), thickness=2)
    # cv.imshow('Rectangle', blank)

    # Draw a Circle
    # cv.circle(blank, (blank.shape[1] // 2, blank.shape[0] // 2), 80, (0, 0, 250), thickness=3)
    # cv.imshow('Circle', blank)

    # Draw a line
    cv.line(blank, (0, 0), (blank.shape[1] // 2, blank.shape[0] // 2), (255, 255, 255), thickness=9)
    cv.imshow('Line', blank)

    cv.waitKey(0)
