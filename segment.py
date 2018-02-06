# libraries
import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise

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


def count_fingers(thresholded, segmented):

    con_hull = cv2.convexHull(segmented)
    # find left right top bottom convex hull points
    extreme_top = tuple(con_hull[con_hull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(con_hull[con_hull[:, :, 1].argmax()][0])
    extreme_left = tuple(con_hull[con_hull[:, :, 0].argmin()][0])
    extreme_right = tuple(con_hull[con_hull[:, :, 0].argmax()][0])
    print(con_hull)
    center_X = (extreme_left[0] + extreme_right[0]) // 2
    center_Y = (extreme_top[0] + extreme_bottom[0]) // 2

    #  find max distance b/w center and a convex_hull point
    distance = pairwise.euclidean_distances([(center_X, center_Y)],
                                            Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]

    # 80% of maximum euclidean distance
    radius = int(0.8 * maximum_distance)

    circumference = (2 * np.pi * radius)

    # take out the circular region coinciding with palm and fingers
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")

    cv2.circle(circular_roi, (center_X, center_Y), radius, 255, 1)

    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # compute the contours in the circular ROI
    (_, contours, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # initalize the finger count
    count = 0

    for c in contours:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # increment the count of fingers only if -
        # 1. The contour region is not the wrist (bottom area)
        # 2. The number of points along the contour does not exceed
        #     25% of the circumference of the circular ROI
        if ((center_Y + (center_Y * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1

    return count


if __name__ == "__main__":
    avg_wt = 0.5
    camera = cv2.VideoCapture(0)
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0

    while True:
        # current frame
        ret_value, frame = camera.read()
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

        if num_frames < 30:
            running_avg(gray, avg_wt)

        else:
            hand = segment(gray)

            if hand is not None:
                # i.e. if segmented
                (thresholded, segmented) = hand

                # draw segment region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                fingers = count_fingers(thresholded, segmented)
                cv2.putText(clone, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
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

    camera.release()
    cv2.destroyAllWindows()
