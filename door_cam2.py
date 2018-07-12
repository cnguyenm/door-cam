"""
An import problem we need to solve for this is that
when we run the face detection,
we somehow need to determine which of the detected faces match
the faces we are already tracking with the correlation trackers.
One simple approach for checking if a detected face matches an existing
correlation tracker region is to see whether the center point of the
detected face is within the region of a tracker
AND if the center point of that same tracker is also within the bound of the detected face.

So the approach to detect and track multiple faces is to use the following steps within our main loop:

    Update all correlation trackers and remove all trackers
    that are not considered reliable anymore (e.g. too much movement)

    Every 10 frames, perform the following:
        1. Use face detection on the current frame and find all faces
        2. For each found face, check whether faces & tracker match (like above)
        if no such tracker exists, we are dealing with a new face
        and we have to start a new tracker for this face.

        3.Use the region information for all trackers
        (including the trackers for the new faces created in the previous step)
        to draw the bounding rectangles

"""

import cv2
import dlib # python deep-learning lib
import threading
import time
import pickle
import argparse

# global
FACE_SIZE = 165
face_recognizer = None 
name_list = []
face_names = {}
face_prop = {}

# init face cascade
faceCascade = cv2.CascadeClassifier('cv_train/haarcascade_frontalface_default.xml')

# desired output width, height
WIDTH = 600
#HEIGHT = 600

def init(name_file, db_file):
    """
    init global variable
    """

    global name_list
    global face_recognizer

    name_list = None
    try:
        with open(name_file, 'rb') as fp:
            name_list = pickle.load(fp) 
    except FileNotFoundError:
        print("[ERROR] cannot read file_name")
        exit(0)

    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    try:
        face_recognizer.read(db_file)
    except cv2.error as e:
        print("[ERROR] cannot read file_db")
        exit(0)

def draw_rect(img, rect, text):
    """
    Draw rect around faces, and write their name
    """
    # draw rect 
    (x, y, w, h) = rect 
    cv2.rectangle(
        img, (x, y), (x+w, y+h), 
        (0, 255, 0), 2
    )

    # write name
    cv2.putText(
        img, text, (x,y-8), cv2.FONT_HERSHEY_PLAIN, 1.5, 
        (0, 255, 0), 2
    )

# we are not really doing face recognition
def recognize_person(face, fid):
    """
    based on faceId, find person name
    add name to faceNames list
    :param faceNames: string list
    :param fid: int
    :return: None
    """

    # specify global var
    global face_names
    global face_recognizer
    
    # convert normal_face to gray_face
    # because I train faces in gray :-|
    # and because it is EigenFace, it requires fixed SIZE
    # which I choose 165, why: it's a nice number
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray_face = cv2.resize(gray_face, (FACE_SIZE, FACE_SIZE))
    label, confidence = face_recognizer.predict(gray_face)
    name = name_list[label]

    # assign name
    face_names[ fid ] = name

    # somehow this thing always return 6555.24234 
    # or sth like that
    face_prop[ fid ] = str(int(confidence/100)) + "%"

def main():

    # parse arg
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--database",
                default='face_recog_opencv/db4.yml',
                help="training data, in yml")
    ap.add_argument("-n", "--name",
                default='face_recog_opencv/name4.pickle',
                help="file name, in pickle")
    args = vars(ap.parse_args())

    # init
    print("[INFO] init")
    global face_names
    init(args['name'], args['database'])

    # open cam device
    # compute output width, height
    cam = cv2.VideoCapture(0)
    # w = cam.get(cv2.CAP_PROP_FRAME_WIDTH) # or: w = cam.get(3)
    # h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT) # or: h = cam.get(4)
    # w2 = WIDTH
    # r  = w2/w 
    # h2 = int(h * r)

    # var holding cur frame, cur faceId
    frame_counter = 0
    cur_face_id = 0

    # var holding correlation trackers, and name per faceId
    face_trackers = {} # fid => tracker
    face_names = {}

    try:
        while True:

            # retrieve latest img from webcam
            time_start = time.time()
            ok, base_img = cam.read()

            # resize img to 320x240
            # base_img = cv2.resize(full_size_base_img, (w2, h2))

            # check if a key press, if it was q, destroy all
            # windows
            k = cv2.waitKey(2)
            if k == ord('q'):
                break


            # STEPS:
            # * Update all trackers and remove the ones
            #   that are not relevant anymore
            # * Every 10 frames:
            #       + Use face detection on cur frame
            #           and look for faces
            #       + For each found face, check if centerpoint
            #           is within existing tracked box. If so,
            #           do nothing
            #       + If centerpoint is NOT in existing tracked box,
            #           then we add a new tracker with a new face-id



            # increase frame counter
            frame_counter += 1


            # update all trackers and remove the ones for which
            # the update indicated the quality was not good enough
            fids_to_delete = []
            for fid in face_trackers.keys():
                tracking_quality = face_trackers[fid].update(base_img)

                # if track quality is NOT good enough, we
                # must del this tracker
                if tracking_quality < 7:
                    fids_to_delete.append(fid)

            for fid in fids_to_delete:
                print("Remove fid: " + str(fid) + " from list of trackers")
                face_trackers.pop(fid, None)



            # Every 10 frames, determine which faces
            # are present in frame
            if (frame_counter % 10) == 0:

                # for face detection, use gray color image
                gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)

                # now, use haar cascade to find all faces
                # in image
                faces = faceCascade.detectMultiScale(gray, 1.3, 5)

                # loop over all faces
                for (_x, _y, _w, _h) in faces:
                    x = int(_x)
                    y = int(_y)
                    w = int(_w)
                    h = int(_h)

                    # cal centerpoint
                    x_bar = x + 0.5*w
                    y_bar = y + 0.5*h

                    # var holding infor which faceId we matched with
                    match_fid = None

                    # now loop over all trackers and check if
                    # center point of the face is within
                    # box of a tracker
                    for fid in face_trackers.keys():
                        track_position = face_trackers[fid].get_position()

                        t_x = int(track_position.left())
                        t_y = int(track_position.top())
                        t_w = int(track_position.width())
                        t_h = int(track_position.height())

                        # cal centerpoint
                        t_x_bar = t_x + 0.5*t_w
                        t_y_bar = t_y + 0.5*t_h

                        # check if center point of face is within
                        # the rect of a tracker region.
                        # Also, the centerpoint of tracker region must
                        # be within the region detected as a face
                        # if both of these conditions hold
                        # we have a match
                        if (
                            # center of face
                            (t_x <= x_bar <= (t_x + t_w)) and
                            (t_y <= y_bar <= (t_y + t_h)) and
                            # center of track
                            (x <= t_x_bar <= (x + w)) and
                            (y <= t_y_bar <= (y + h))
                        ):
                            match_fid = fid

                    # if no matched fid, then have to create a new tracker
                    if match_fid is None:
                        print("create a new tracker " + str(cur_face_id))

                        # create and store the tracker
                        tracker = dlib.correlation_tracker()
                        tracker.start_track(base_img,
                                            dlib.rectangle(
                                                x - 10,
                                                y - 20,
                                                x + w + 10,
                                                y + h + 20
                                            ))
                        face_trackers[cur_face_id] = tracker

                        # start a new thread, to simulate face recognition
                        
                        t = threading.Thread(
                            target= recognize_person,
                            args = (base_img[y:y+h, x:x+w], cur_face_id)
                        )
                        t.start()

                        # increase curfaceId counter
                        cur_face_id += 1

            # end 10 frame-if

            # now loop over all trackers and raw rect
            # around detected faces.
            for fid in face_trackers.keys():
                track_position = face_trackers[fid].get_position()

                t_x = int(track_position.left())
                t_y = int(track_position.top())
                t_w = int(track_position.width())
                t_h = int(track_position.height()) 
                rect = (t_x, t_y, t_w, t_h)

                # draw rect, write name (if it exists)
                if fid in face_names:
                    draw_rect(base_img, rect, "{}:{}".format(face_names[fid], face_prop[fid]) )
                else:
                    draw_rect(base_img, rect, "unknown")

            # finally, show images on screen
            time_end = time.time()
            fps = int(1/(time_end-time_start))
            fps_display = "fps: " + str(fps)
            cv2.putText(base_img, fps_display, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('base-image', base_img)

    # if user press Ctrol-C in console
    # break out of main loop
    except KeyboardInterrupt as e:
        pass

    # destroy openCV windows
    cv2.destroyAllWindows()
    exit(0)



if __name__ == '__main__':
    main()


















