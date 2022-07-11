import cv2 as cv
import numpy as np

if __name__ == '__main__':
    img = cv.imread('images/cat.jpg')
    cv.imshow('Cat', img)

    blank = np.zeros(img.shape, dtype='uint8')

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Gray', gray)

    blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
    cv.imshow('Gaussian Blur', blur)

    canny = cv.Canny(blur, 125, 175)
    cv.imshow('Canny Edges', canny)

    ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
    cv.imshow('Thresh', canny)

    contours, hierachies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    print(f'Contours({len(contours)} found!)')

    cv.drawContours(blank, contours, -1, (0, 0, 255), 2)
    cv.imshow('Contours Draw', blank)

    cv.waitKey(0)
