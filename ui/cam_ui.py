import cv2
import time
from face_keeper import FaceKeeper

# const variable
BASE_IMG_SIZE = (320, 240)
LARGE_IMG_SIZE = (775, 600)  # width, height


COLOR_RED   = (0, 0, 255)
COLOR_BROWN = (0, 165, 255)
COLOR_GREEN = (0, 255, 0)


class CamUI:
    def __init__(self, src=0):

        # init camera
        self.cam = cv2.VideoCapture(src)
        cv2.namedWindow('base-img', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('result-img', cv2.WINDOW_AUTOSIZE)

        # position the windows next to each other
        cv2.moveWindow('base-img', 0, 100)  # x, y
        cv2.moveWindow('result-img', 400, 100)

        # start window thread for 2 windows
        cv2.startWindowThread()

        # var
        self.face_keeper = FaceKeeper()

    def draw(self):
        cv2.imshow('base-img', self.base_img)
        cv2.imshow('result-img', self.result_img)

    def run(self):
        try:
            frame_counter = 0

            while True:

                # retrieve latest img from cam
                time_start = time.time()
                ok, full_size_base_img = self.cam.read()

                # resize base_img smaller
                base_img = cv2.resize(full_size_base_img, BASE_IMG_SIZE)
                result_img = base_img.copy()

                # if `q` key is pressed, quit
                k = cv2.waitKey(2)
                if k == ord('q'):
                    break

                # every frame, update tracker
                self.face_keeper.update_tracker(base_img)

                # every 10 frames, detect face, recognize names
                if frame_counter % 10 == 0:
                    self.face_keeper.detect_face(base_img)

                # now loop over all trackers and draw rect, names
                # around detected faces.
                self.face_keeper.draw_face_rect(result_img)

                # create large result
                large_result = cv2.resize(result_img, LARGE_IMG_SIZE)

                # cal fps
                time_end = time.time()
                fps = int(1 / (time_end - time_start))
                fps_display = "fps: " + str(fps)
                cv2.putText(large_result, fps_display,
                            (0, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, COLOR_RED, 2)

                # finally draw images on screen
                cv2.imshow('base-img', base_img)
                cv2.imshow('result-img', large_result)

        except KeyboardInterrupt as e:
            pass

        cv2.destroyAllWindows()
        quit(0)


def main():
    print("this file cannot run alone")


if __name__ == '__main__':
    CamUI(src="video/learn_cat.mp4").run()
