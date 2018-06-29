"""
Same as detect_and_track_multiple
However, use face_recognition to detect face
instead of using opencv haarcascade

haar cascade => face_recognition
KCF_tracker => dlib correlation tracker
detect 1 per frame => detect 10 frame

improve performance + accuracy

"""

import cv2
import dlib  # python deep-learning lib
import threading
import time
import face_recognition
import argparse
import pickle


# desired output width, height
OUTPUT_SIZE_WIDTH = 775
OUTPUT_SIZE_HEIGHT = 600
COLOR_RED   = (0, 0, 255)
COLOR_BROWN = (0, 165, 255)
COLOR_GREEN = (0, 255, 0)


# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
                help="path to serialized db of facial encodings")

ap.add_argument("-i", "--input", required=True,
                help="path to input camera")

# only use 'cnn' with GPU, CUDA
ap.add_argument("-d", "--detection", type=str, default="hog",
                help="detection model: `hog` or `cnn`")


args = vars(ap.parse_args())

# load known faces and embeddings
print("[INFO] loading encodings")
data = pickle.loads(open(args["encodings"], "rb").read())
print("[INFO] load {} encodings from db".format(len(data["encodings"])))


def check_int(s):
    """
    Helper function, check if s is int

    :param s: <string>
    :return: <boolean>
    """
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()


# we are not really doing face recognition
def recognize_person(face_names, fid, rgb_frame, bbox):
    """
    based on faceId, find person name
    add name to faceNames list

    :param face_names: string list
    :param fid: int
    :param rgb_frame: <numpy.ndarray> base_image in RGB format
    :param bbox: list of bboxes, in this case 1
    :return: None
    """

    # if there is n_faces, output: n_encodings
    # in this case, there is only 1 face,
    # so take index 0
    encoding = face_recognition.face_encodings(rgb_frame, bbox)[0]
    name = "Unknown"

    # attempt to match each face in input image to our
    # known encodings
    # tolerance: default=0.6
    # return: list [T, F, T, ...]
    matches = face_recognition.compare_faces(
        data["encodings"], encoding
    )

    # check if we have a match
    if True in matches:
        # find indexes of al matched faces, then
        # init a dict to count the total number of times
        # each face was matched
        matched_ids = [i for (i, b) in enumerate(matches) if b]
        counts = {}

        # loop over the matched indexes
        # maintain a count for each recognized face
        for i in matched_ids:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1

        # determine the recognized face with
        # largest number of votes
        # (if tie, Python will select 1st entry)
        name = max(counts, key=counts.get)

    # assign name to face
    face_names[fid] = name


def detect_and_track_multiple_faces():

    # open cam device
    print("[INFO] Start capture video.")
    if check_int(args["input"]):
        in_cam = int(args["input"])
    else:
        in_cam = args["input"]

    cam = cv2.VideoCapture(in_cam)

    # create 2 opencv named windows
    cv2.namedWindow('base-image', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('result-image', cv2.WINDOW_AUTOSIZE)

    # position the windows next to each other
    cv2.moveWindow('base-image', 0, 100)  # x, y
    cv2.moveWindow('result-image', 400, 100)

    # start window thread for 2 windows
    cv2.startWindowThread()

    # color of rect we draw around face
    rect_color = COLOR_BROWN

    # var holding cur frame, cur faceId
    frame_counter = 0
    cur_face_id = 0

    # var holding correlation trackers, and name per faceId
    face_trackers = {}  # fid => tracker
    face_names = {}

    try:
        while True:

            # retrieve latest img from webcam
            time_start = time.time()
            ok, full_size_base_img = cam.read()

            # resize img to 320x240
            base_img = cv2.resize(full_size_base_img, (320, 240))

            # check if a key press, if it was q, destroy all
            # windows
            k = cv2.waitKey(2)
            if k == ord('q'):
                break

            # result img, the one we show user
            # result = original + rects
            result_img = base_img.copy()

            # STEPS:
            # * Update all trackers and remove the ones
            #   that are not relevant anymore
            # * Every 10 frames:
            #       + Use face detection on cur frame
            #           and look for faces
            #       + For each found face, check if centerpoint
            #           is within existing tracked box. If so,
            #           do nothing
            #       + If center-point is NOT in existing tracked box,
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
                print("[INFO] Remove fid: " + str(fid) + " from list of trackers")
                face_trackers.pop(fid, None)

            # Every 10 frames, determine which faces
            # are present in frame
            if (frame_counter % 10) == 0:

                # convert BGR to RGB because
                # face_recognition use RGB
                rgb_frame = base_img[:, :, ::-1]

                # detect faces
                faces = face_recognition.face_locations(
                    rgb_frame,
                    model=args["detection"]
                )

                # loop over all faces
                # top, right, bottom, left
                for (_yt, _xr, _yb, _xl) in faces:
                    yt = int(_yt)
                    yb = int(_yb)
                    xl = int(_xl)
                    xr = int(_xr)

                    # cal centerpoint
                    x_bar = xl + 0.5*(xr-xl)
                    y_bar = yt + 0.5*(yb-yt)

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

                        # cal center-point
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
                            (xl <= t_x_bar <= xr) and
                            (yt <= t_y_bar <= yb)
                        ):
                            match_fid = fid

                            # if found a match, no need to
                            # find anymore
                            break

                    # if no matched fid, then have to create a new tracker
                    if match_fid is None:
                        print("[INFO] Create a new tracker " + str(cur_face_id))

                        # create and store the tracker
                        tracker = dlib.correlation_tracker()
                        tracker.start_track(base_img,
                                            dlib.rectangle(
                                                xl - 10,
                                                yt - 20,
                                                xr + 10,
                                                yb + 20
                                            ))
                        face_trackers[cur_face_id] = tracker

                        # start a new thread, to recognize face
                        bbox = [(yt, xr, yb, xl)]
                        t = threading.Thread(
                            target=recognize_person,
                            args=(face_names, cur_face_id,
                                  rgb_frame, bbox)
                        )
                        t.start()

                        # increase cur_faceId counter
                        cur_face_id += 1

            # end 10 frame-if

            # now loop over all trackers and draw rect
            # around detected faces.
            for fid in face_trackers.keys():
                track_position = face_trackers[fid].get_position()

                t_x = int(track_position.left())
                t_y = int(track_position.top())
                t_w = int(track_position.width())
                t_h = int(track_position.height())

                cv2.rectangle(result_img,
                              (t_x, t_y),
                              (t_x + t_w, t_y + t_h),
                              rect_color, 2)

                # write name ...
                if fid in face_names:
                    cv2.putText(result_img, face_names[fid],
                                (t_x, t_y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                COLOR_GREEN, 2)

            # create large result
            large_result = cv2.resize(result_img,
                                      (OUTPUT_SIZE_WIDTH, OUTPUT_SIZE_HEIGHT))

            # finally, show images on screen
            time_end = time.time()
            fps = int(1/(time_end-time_start))
            fps_display = "fps: " + str(fps)
            cv2.putText(large_result, fps_display,
                        (0, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, COLOR_RED, 2)

            cv2.imshow('base-image', base_img)
            cv2.imshow('result-image', large_result)

    # if user press Ctrol-C in console
    # break out of main loop
    except KeyboardInterrupt as e:
        pass

    # destroy openCV windows
    print("[INFO] Exit.")
    cv2.destroyAllWindows()
    exit(0)


if __name__ == '__main__':
    detect_and_track_multiple_faces()



















