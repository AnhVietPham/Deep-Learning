import cv2 as cv

if __name__ == '__main__':
    img = cv.imread('images/cat.jpg')
    cv.imshow('Cat', img)

    # Averaging
    average = cv.blur(img, (25, 25))
    cv.imshow('average Cat', average)

    # Gaussian Bluring
    gaussian_blur = cv.GaussianBlur(img, (25, 25), 0)
    cv.imshow('Gaussian Blur', gaussian_blur)

    # Median
    median = cv.medianBlur(img, 25)
    cv.imshow('Median Image', median)

    # Bilateral
    bilateral = cv.bilateralFilter(img, 17, 112, 112)
    cv.imshow('Bilateral Image', bilateral)
    cv.waitKey(0)
