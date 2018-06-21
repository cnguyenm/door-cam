"""
Detect a face, and track it
"""

import cv2
import dlib # python deep-learning lib

# init face cascade
faceCascade = cv2.CascadeClassifier('cv_train/haarcascade_frontalface_default.xml')

# create tracker
tracker = dlib.correlation_tracker()

# var to know whether we are currently using dlib tracker
tracking_face = 0

# desired output width, height
OUTPUT_SIZE_WIDTH = 775
OUTPUT_SIZE_HEIGHT = 600

# open first webcam device
cam = cv2.VideoCapture(1)

# create 2 opencv windows
cv2.namedWindow('base-image', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('result-image', cv2.WINDOW_AUTOSIZE)

# position windows next to each other
cv2.moveWindow('base-image', 0, 100)
cv2.moveWindow('result-image', 400, 100)

# start window thread for 2 windows we are using
cv2.startWindowThread()

rectColor = (0, 165, 255)

# inf loop
while True:

    # retrieve latest image from webcam
    rc, full_size_base_img = cam.read()

    # resize img to 320x240
    base_img = cv2.resize(full_size_base_img, (320, 240))

    # check if a key press, if it was q, destroy all
    # windows
    k = cv2.waitKey(2)
    if k == ord('q'):
        cv2.destroyAllWindows()
        exit(0)

    # result img = original img + rect for face
    result_img = base_img.copy()

    # if we are not tracking a face, try to detect one
    if not tracking_face:

        # for face detection, use a gray colored img
        gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)

        # use Haar cascade to find all faces
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)

        # for now, we are only interested in 'largest' face,
        # determine this based on largest area of found rect
        # first, init var to 0
        max_area = 0
        x = 0
        y = 0
        w = 0
        h = 0

        # loop over all faces, find largest rect so far
        for (_x, _y, _w, _h) in faces:
            if _w * _h > max_area:
                x, y, w, h = _x, _y, _w, _h
                max_area = w * h

        # draw a rect around the largest face in picture
        if max_area > 0:

            # init dlib tracker
            tracker.start_track(base_img,
                                dlib.rectangle(x-10,
                                               y-20,
                                               x+w+10,
                                               y+h+20))

            # set indicator
            tracking_face = 1


    # if tracker is actively tracking a region
    if tracking_face:

        # update  tracker, request information
        # about quality of tracking update
        tracking_quality = tracker.update(base_img)

        # if tracking quality is good enough, determine
        # the updated position of tracked region and
        # draw the rectangle
        if tracking_quality >= 8.75:
            tracked_position = tracker.get_position()

            t_x = int(tracked_position.left())
            t_y = int(tracked_position.top())
            t_w = int(tracked_position.width())
            t_h = int(tracked_position.height())
            cv2.rectangle(result_img, (t_x, t_y),
                          (t_x + t_w, t_y + t_h),
                          rectColor, 2)

        # if quality of tracking update is enough
        # stop tracking
        else:
            tracking_face = 0




    # show sth larger on the screen than original 320x240
    # resize img again
    large_result = cv2.resize(result_img,
                              (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT))

    # finally, show images on screen
    cv2.imshow('base-image', base_img)
    cv2.imshow('result-image', large_result)
























