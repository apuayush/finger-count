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
    (_, cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPPROX_SIMPLE)

    if len(cnts) == 0:
        # for no contour
        return
    else:
        # maximum contour area
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

