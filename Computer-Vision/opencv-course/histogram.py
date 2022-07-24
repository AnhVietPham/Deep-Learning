from __future__ import print_function
import cv2 as cv
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img = cv.imread('images/dog1.jpeg')
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(gray_img)
    gray_hst = cv.calcHist([gray_img], [0], None, [256], [0, 256])
    gray_hst_dst = cv.calcHist([dst], [0], None, [256], [0, 256])
    cv.imshow('Source image', img)
    cv.imshow('Source gray image', dst)
    cv.imshow('Equalized Image', gray_img)

    plt.figure()
    plt.title("GrayScale Histogram")
    plt.xlabel('Bins')
    plt.ylabel('# of pixels')
    plt.plot(gray_hst_dst)
    plt.xlim([0, 255])
    plt.show()
    cv.waitKey()
