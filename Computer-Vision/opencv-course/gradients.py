import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    img = cv.imread('images/cat.jpg')
    # cv.imshow('Cat', img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("Gray Image", gray)

    # Laplacian
    #  we deduce that the second derivative can be used to detect edges. Since images are "*2D*", we would need to take
    #  the derivative in both dimensions. Here, the Laplacian operator comes handy.
    lap = cv.Laplacian(gray, cv.CV_64F)
    lap = np.uint8(np.absolute(lap))
    cv.imshow("Laplacian", lap)

    # Sobel
    sobelX = cv.Sobel(gray, cv.CV_64F, 1, 0)
    sobelY = cv.Sobel(gray, cv.CV_64F, 0, 1)
    combined_sobel = cv.bitwise_or(sobelX, sobelY)

    cv.imshow("Sobel X", sobelX)
    cv.imshow("Sobel Y", sobelY)
    cv.imshow("Combined Sobel", combined_sobel)

    cv.waitKey(0)
