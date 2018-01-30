# libraries
import cv2
import imutils
import numpy as np

bg = None


def running_avg(img, avg_wt):
    global bg

    # for the first time
    if bg is None:
        bg = img.copy().astype('float')
        return

    # accumulating weighted average and updating the background
    cv2.accumulateWeighted(img, bg, avg_wt)


def segment(img, threshold=25):
    global bg
    # absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), img)

    # to get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # contours
    (_, cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        # for no contour
        return
    else:
        # maximum contour area
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)


if __name__ == "__main__":
    avg_wt = 0.5
    camera = cv2.VideoCapture(0)
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0

    while True:
        # current frame
        ret_value, frame = camera.read()
        print(ret_value)

        frame = imutils.resize(frame, width=700)
        # flip the frame to remove mirror view
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        # get the height and width of the frame
        (height, width) = frame.shape[:2]
        # get the ROI
        roi = frame[top:bottom, right:left]
        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # calibrating our running average until a threshold is reached

        if num_frames<30:
            running_avg(gray, avg_wt)

        else:
            hand = segment(gray)

            if hand is not None:
                # i.e. if segmented
                (thresholded, segmented) = hand

                # draw segment region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)

        # draw segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

    # free up memory
    camera.release()
    cv2.destroyAllWindows()


    #
    #
    #
    #
