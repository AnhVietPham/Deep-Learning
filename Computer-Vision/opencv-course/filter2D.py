import cv2 as cv
import numpy as np
from rescale import rescaleFrame

if __name__ == '__main__':
    img = cv.imread('images/opencv.png')
    cv.imshow('Open CV', img)

    # Blur an image with a 2d convolution matrix
    kernelBlurImage = np.ones((5, 5), np.float32) / 25
    blurImage = cv.filter2D(img, -1, kernelBlurImage)
    cv.imshow('Blur Image Open CV2', blurImage)

    # Edge detect an image with a 2d convolution matrix
    kernelEdgeDetect = np.array([[-1, -1, -1],
                                 [-1, 8, -1],
                                 [-1, -1, -1]])
    edgeDetectImage = cv.filter2D(img, -1, kernelEdgeDetect)
    cv.imshow('Edge Detect Image Open CV2', edgeDetectImage)

    cv.waitKey(0)
