import cv2 as cv
import numpy as np

if __name__ == '__main__':
    blank = np.zeros((400, 400), dtype='uint8')

    rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)
    cv.imshow("Rectangel", rectangle)

    circle = cv.circle(blank.copy(), (200, 200), 200, 255, -1)
    cv.imshow("Circle", circle)

    # Bitwise And
    bitwsise_and = cv.bitwise_and(circle, rectangle)
    cv.imshow('Bitwise And', bitwsise_and)

    # Bitwise Or
    bitwise_or = cv.bitwise_or(circle, rectangle)
    cv.imshow('Bitwise Or', bitwise_or)

    # Bitwise XOR
    bitwise_xor = cv.bitwise_xor(rectangle, circle)
    cv.imshow('Bitwise XOr', bitwise_xor)

    cv.waitKey(0)
