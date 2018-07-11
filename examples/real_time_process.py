import numpy as np
import cv2
import os
import time
import sys, getopt
from video import create_capture
from common import clock, draw_str


def run_cam_real_time_opencv():


    # ----- Load Detector ----------------------------------------------------
    cwd = os.path.dirname(os.path.realpath(__file__))
    haarcascade = os.path.join(cwd, "cv_train/haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(haarcascade)


    # ----- Font ----------------------------------------------------
    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (0, 255, 0)  # BGR
    topleft = (0, 15)

    # ----- Create capture -----------------------------------------------------
    # cam = create_capture("videos/SweetDream.mp4")
    cam = cv2.VideoCapture(0)  # wrapper for cv2.VideoCapture
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    # ----- While loop -----------------------------------------------------
    while True:

        # read frame
        start = time.time()
        ret, frame = cam.read()  # ret = True if there is still image
        #frame = cv2.flip(frame, 1) # turn camera into a mirror

        # detect face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # draw bbox
        draw_boxes(frame, faces)

        # cal fps
        end = time.time()
        fps = int(1 / (end - start))
        display = "fps: " + str(fps)

        # render video
        cv2.putText(frame, display, topleft, font, 0.5, white, 2, cv2.LINE_AA)
        cv2.imshow('Demo', frame)

        # exit condition
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

def draw_boxes(img, detections):
    """
    Draw bounding boxes around person detected
    for openCV haarcascade
    :param img: cv2.image
    :param detections: detections return from openCV
    :return: None
    """

    # ----- While loop: drawing -----------------------------------------------------
    for detect in detections:

        # get coordinates
        (x, y, w, h) = detect
        top_left     = ( int(x) , int(y))
        bottom_right = ( int(x+w) , int(y+h))
        color        = (0, 255, 0) # green
        thick        = 2

        # draw
        cv2.rectangle(img, top_left, bottom_right, color=color, thickness=thick)

def test_tracker():

    # set up tracker
    tracker_types = ['BOOSTING', 'MIL', 'KCF'
                     , 'TLD', 'MEDIANFLOW']
    tracker_type = tracker_types[2]

    tracker = cv2.TrackerKCF_create()

    # Define an initial bounding box
    bbox = (0, 0, 50, 100)

    # read video
    cam = cv2.VideoCapture(0)
    ok, frame = cam.read()
    bbox = cv2.selectROI(frame, showCrosshair=True)
    ok = tracker.init(frame, bbox)

    while True:
        ok, frame = cam.read()

        # update tracker
        ok, bbox = tracker.update(frame)

        # draw bbox
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Display tracker type on frame
            # no need

        cv2.imshow("Tracking", frame)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

# run run run
test_tracker()































