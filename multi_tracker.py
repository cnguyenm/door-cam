import cv2 as cv
import common
import time

class TrackedObject():
    """
    Help to store data when track object
    """
    def __init__(self, id, tracker):
        self.id = id
        self.tracker = tracker

class CustomMultiTracker():
    """
    For each object, use KCFtracker to track
    object
    """
    def __init__(self):
        self.track_list = []

    def add_object(self, frame, bbox):
        """
        add object to track to track_list
        :param frame: init frame
        :param bbox:  init bbox
        :return: True if add successfully
            False otherwise
        """
        # create tracker
        tracker = cv.TrackerKCF_create()
        ok = tracker.init(frame, bbox)
        if (not ok):
            return False

        # create TrackedObject
        id = len(self.track_list)

        # add to track list
        self.track_list.append((id, tracker))
        return True

    def update(self, frame):
        """
        update each tracker in list
        if an object lost track, remove that object
        :return:
        [ (id, bbox),
          (id2, (x, y, w, h)) ]
        """

        # return value
        result = []

        # copy track_list so we can modify
        # original list
        for t in self.track_list:

            # update each tracker
            ok, bbox = t[1].update(frame)

            # if not ok, remove object
            if (not ok):
                self.track_list.remove(t)

            # if ok, add to list
            result.append((t[0], bbox))

        return result

    def clear(self):
        """
        Clear track list, remove
        all references as well
        """
        del self.track_list[:]


class App():
    def __init__(self):
        self.cam = cv.VideoCapture(0)
        self.frame = None
        self.paused = False
        self.multi_tracker = CustomMultiTracker()
        self.tracker = cv.TrackerKCF_create()
        self.isTracking = False

        cv.namedWindow("demo")
        self.rect_sel = common.RectSelector('demo',self.on_rect)

    def on_rect(self, rect):
        """
        Called when user drag a rect on video
        by common.RectSelector
        """
        #self.multi_tracker.add_object(self.frame, rect)
        self.tracker.init(self.frame, rect)
        self.isTracking = True

    def draw_boxes(self, img, detections):
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
            (x, y, w, h) = detect[1]
            top_left = (int(x), int(y))
            bottom_right = (int(x + w), int(y + h))
            color = (0, 255, 0)  # green
            thick = 2

            # draw
            cv.rectangle(img, top_left, bottom_right, color, thick, 1)

    def run(self):
        while True:
            # see if cam is playing
            playing = not self.paused and \
                    not self.rect_sel.dragging

            # if playing
            if playing or self.frame is None:

                # read frame
                t1 = time.time()
                ret, frame = self.cam.read()
                if not ret:
                    break

                self.frame = frame.copy()

            # copy frame to display
            # so we don't mess up original
            vis = self.frame.copy()
            display_text = ""

            # if not dragging and playing
            # update track
            if playing and self.isTracking:
                #tracked_objects = self.multi_tracker.update(frame)
                ok, bbox = self.tracker.update(frame)
                #display_text += "track objects: " + str(len(tracked_objects))
                #self.draw_boxes(vis, tracked_objects)
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv.rectangle(vis, p1, p2, (255, 0, 0), 2, 1)
            else:

                # if is selecting rect
                self.rect_sel.draw(vis)
                t2 = time.time()
                if (t2 - t1 == 0):
                    fps = 0
                else:
                    fps = int(1 / (t2 - t1))
                fps_display = "fps: " + str(fps)
                # render frame
                cv.putText(vis, display_text, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2 )
                cv.putText(vis, fps_display, (0, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv.imshow('demo', vis)

                k = cv.waitKey(1)
                if k == 27:
                    break

        # destroy all window to free up mem
        cv.destroyAllWindows()

App().run()















