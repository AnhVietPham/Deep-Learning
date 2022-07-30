import numpy as np
import cv2 as cv

if __name__ == '__main__':
    img = cv.imread('images/chessboard.png')
    cv.imshow('Image', img)

    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    gray = np.float32(gray_image)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    # dst = cv.dilate(dst, None)
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst < 0.01 * dst.max()] = [0, 0, 255]
    cv.imshow('dst', img)

    cv.waitKey(0)
