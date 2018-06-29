import cv2
import face_recognition
import dlib
import time
import threading

COLOR_RED   = (0, 0, 255)
COLOR_BROWN = (0, 165, 255)
COLOR_GREEN = (0, 255, 0)


class FaceKeeper:
    def __init__(self, detect_model="hog"):
        self.face_names     = {}  # dict of <fid, names>
        self.face_trackers  = {}  # dict of <fid, dlib.correlation trackers>
        self.cur_face_id    = 0

        # 'hog' or 'cnn',
        # 'cnn' should only be used with gpu, cuda
        self.detect_model   = detect_model

    def update_tracker(self, base_img):
        """
        update all trackers, and remove the ones for which
        the update indicated the quality was not good enough

        :param base_img: <numpy.ndarray> small size base_img
        :return: None
        """
        fids_to_delete = []
        for fid in self.face_trackers.keys():
            track_quality = self.face_trackers[fid].update(base_img)

            # if track quality is NOT good enough,
            # del this tracker
            if track_quality < 7:
                fids_to_delete.append(fid)

        for fid in fids_to_delete:
            print("[INFO][FaceKeeper] Remove fid: {} ".format(fid))

            # update tracker
            self.face_trackers.pop(fid, None)

            # update names
            if fid in self.face_names:
                self.face_names.pop(fid, None)

    def detect_face(self, base_img):
        """
        Detect face + detect name

        :param base_img: <numpy.ndarray> small size base_img
        :return:
        """
        # convert BGR to RGB because
        # face_recognition use RGB
        rgb_frame = base_img[:, :, ::-1]

        # detect faces
        faces = face_recognition.face_locations(
            rgb_frame,
            model=self.detect_model
        )

        # loop over all faces
        # top, right, bottom, left
        for (_yt, _xr, _yb, _xl) in faces:

            # convert to int, because face recognition may return float
            yt, yb, xl, xr = int(_yt), int(_yb), int(_xl), int(_xr)

            # check if there is any tracker for that face
            match_fid = self.get_track_id_from_location((yt, xr, yb, xl))

            # if No tracker yet
            if match_fid is None:
                print("[INFO][FaceKeeper] Create a new tracker {}".format(self.cur_face_id))

                # create and store the tracker
                tracker = dlib.correlation_tracker()
                tracker.start_track(base_img,
                                    dlib.rectangle(
                                        xl - 10,
                                        yt - 20,
                                        xr + 10,
                                        yb + 20
                                    ))
                self.face_trackers[self.cur_face_id] = tracker

                # start a new thread to recognize face
                # because it may take some time
                t = threading.Thread(
                    target=self.recognize_face,
                    args=(self.cur_face_id, rgb_frame)
                )

                t.start()

                # increase current_face_id
                self.cur_face_id += 1

    def get_track_id_from_location(self, face_loc):
        """
        given face_location, loop over all trackers
        check if any tracker and face overlap.
        If yes, return that trackId, which is also faceId

        :param face_loc: (top, right, bottom, left) face location
        :return: <int> tracker id
                or None if not found
        """
        yt, xr, yb, xl = face_loc

        # calculate  center point
        x_mid = xl + 0.5 * (xr - xl)
        y_mid = yt + 0.5 * (yb - yt)

        # matched faceId
        match_fid = None

        # now loop over all trackers
        for fid in self.face_trackers.keys():
            track_position = self.face_trackers[fid].get_position()

            t_x = int(track_position.left())
            t_y = int(track_position.top())
            t_w = int(track_position.width())
            t_h = int(track_position.height())

            # calculate tracker center point
            t_x_mid = t_x + 0.5 * t_w
            t_y_mid = t_y + 0.5 * t_h

            # check if
            # face_center is within tracker_box
            # and tracker_center is within face_box
            if (
                    # center of face
                    (t_x <= x_mid <= (t_x + t_w)) and
                    (t_y <= y_mid <= (t_y + t_h)) and
                    # center of track
                    (xl <= t_x_mid <= xr) and
                    (yt <= t_y_mid <= yb)
            ):
                match_fid = fid

                # if found a match, no need to
                # find anymore
                break

        return match_fid

    def recognize_face(self, fid, rgb_frame):
        """

        :param fid: int
        :return:
        """
        time.sleep(2)
        self.face_names[fid] = "Person " + str(fid)

    def draw_face_rect(self, result_img):
        # now loop over all trackers and draw rect
        # around detected faces.
        for fid in self.face_trackers.keys():
            track_position = self.face_trackers[fid].get_position()

            t_x = int(track_position.left())
            t_y = int(track_position.top())
            t_w = int(track_position.width())
            t_h = int(track_position.height())

            # draw rect
            cv2.rectangle(result_img,
                          (t_x, t_y),
                          (t_x + t_w, t_y + t_h),
                          COLOR_BROWN, 2)

            # write name ...
            if fid in self.face_names:
                cv2.putText(result_img, self.face_names[fid],
                            (t_x, t_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            COLOR_GREEN, 1)
















