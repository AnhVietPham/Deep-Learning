import cv2 as cv


def rescaleFrame(frame, scale=2):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimension = (width, height)
    return cv.resize(frame, dimension, interpolation=cv.INTER_AREA)


def rescaleVideo():
    capture = cv.VideoCapture('videos/video1.mp4')
    while True:
        isTrue, frame = capture.read()
        frame_resized = rescaleFrame(frame)
        cv.imshow('Video', frame)
        cv.imshow('Video Resized', frame_resized)

        if cv.waitKey(20) & 0xFF == ord('d'):
            break
    capture.release()
    cv.destroyAllWindows()


def rescaleImage():
    img = cv.imread('images/cat.jpg')
    img_resized = rescaleFrame(img)
    cv.imshow('Cat', img)
    cv.imshow('Cat Resized', img_resized)
    cv.waitKey(0)


if __name__ == '__main__':
    rescaleImage()
