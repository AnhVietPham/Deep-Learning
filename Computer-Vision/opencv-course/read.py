import cv2 as cv


def showImage():
    img = cv.imread('images/cat.jpg')
    cv.imshow('Cat', img)
    cv.waitKey(0)


def playVideo():
    capture = cv.VideoCapture('videos/video1.mp4')
    while True:
        isTrue, frame = capture.read()
        cv.imshow('Video', frame)

        if cv.waitKey(20) & 0xFF == ord('d'):
            break
    capture.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    playVideo()
