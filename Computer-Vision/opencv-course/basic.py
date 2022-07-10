import cv2 as cv

if __name__ == '__main__':
    img = cv.imread('images/cat.jpg')
    cv.imshow('Cat', img)

    # Converting to Grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow('Gray', gray)

    # Blur
    blur = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT)
    # cv.imshow('Blur', blur)

    # Edge Cascade
    canny = cv.Canny(blur, 125, 175)
    cv.imshow('Canny', canny)

    cv.waitKey(0)
