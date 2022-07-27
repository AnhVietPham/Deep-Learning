import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv.imread('images/cat.jpg')
    # cv.imshow('Cat', img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow("Gray Image", gray)

    # Simple Thresholding
    ret, thresh1 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    ret2, thresh2 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)
    ret3, thresh3 = cv.threshold(gray, 127, 255, cv.THRESH_TRUNC)
    ret4, thresh4 = cv.threshold(gray, 127, 255, cv.THRESH_TOZERO)
    ret5, thresh5 = cv.threshold(gray, 127, 255, cv.THRESH_TOZERO_INV)
    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    gray_images = [gray, thresh1, thresh2, thresh3, thresh4, thresh5]

    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(gray_images[i], 'gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

    # Adaptive Thresholding
    # ret, th1 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    # th2 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    # th3 = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    # titles = ['Original Image', 'Global Thresholding (v = 127)',
    #           'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    # gray_images = [gray, th1, th2, th3]
    # for i in range(4):
    #     plt.subplot(2, 2, i + 1), plt.imshow(gray_images[i], 'gray')
    #     plt.title(titles[i])
    #     plt.xticks([]), plt.yticks([])
    # plt.show()

    # cv.waitKey(0)
