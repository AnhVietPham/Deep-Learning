import cv2 as cv

if __name__ == '__main__':
    img = cv.imread('images/cat.jpg')
    cv.imshow('Cat', img)

    # BGR to GrayScale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Gray Image', gray)

    # BGR to HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    cv.imshow('HSV Image', hsv)

    # BGR to LAB
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    cv.imshow('LAB Image', lab)

    cv.waitKey()
