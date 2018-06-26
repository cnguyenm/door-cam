
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

# construct arg parser
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
                help="path to db of face encodings")

ap.add_argument("-o", "--output", type=str,
                help="path to output video")

ap.add_argument("-y", "--display", type=int, default=1,
                help="set 1 to display output frame to screen")

ap.add_argument("-d", "--detection", type=str, default="hog",
                help="face detection model to use: `hog` or `cnn`")

args = vars(ap.parse_args())

# load known faces, and embeddings
print("[INFO] loading encodings") if __debug__ else 0
data = pickle.loads(open(args["encodings"], "rb").read())

# init video stream and pointer to output video file
# VideoStream: wrapper class for cv2.Video
# diff: has a separate thread running
print("[INFO] start video stream") if __debug__ else 0
video = VideoStream(src=0).start()
writer = None
frame_counter = 0
time.sleep(2.0)

# Steps:
# 1. Read each frame
# 2. for every 10 frame, perform recognize face
# 3. If display => display
# 4. If write => write
while True:
    # ---- process each frame from video ------------------
    t1 = time.time()
    frame = video.read()

    # convert frame BGR<=>RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(rgb.shape[1]) # r = proportion

    # detect faces
    boxes = face_recognition.face_locations(
        rgb,
        model=args["detection"]
    )

    # encode faces: each face=>encoding
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # step:
    # 1.foreach face found, compare that face with known db
    # 2.then choose face that have most matches
    # 3.if no match, leave name as Unknown
    for e in encodings:

        # attempt to match each face in input image to our known
        # encodings
        matches = face_recognition.compare_faces(
            data["encodings"], e
        )
        name = "Unknown"

        # check to see if we have found a match
        if True in matches:

            # find indexes of all matched faces, then init
            # a dict to count the total number of times
            # each face was matched
            # ex: Ids = [3, 5, 7, 10, 11, 12]
            # encoding 3,5 match the current face
            matchedIds = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over matched index and maintain a count
            # for each recognized face
            for i in matchedIds:
                # data["names"] = list ["joe", "joe", "eve", ...]
                name = data["names"][i]

                # counts = {"joe":2, "eve":5}
                # so result = "eve"
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with largest number
            # of votes
            name = max(counts, key=counts.get)

        # update list of names
        # name="unknown"or "name_of_max_votes"
        names.append(name)

    # ---- draw boxes around recognized faces ------------------
    for ((y_t, x_r, y_b, x_l), name) in zip(boxes, names):
        # rescale face from rgb => frame
        # r = frame/rgb
        # => frame = r * rgb
        y_t = int(y_t * r)
        y_b = int(y_b * r)
        x_r = int(x_r * r)
        x_l = int(x_l * r)

        # draw box
        cv2.rectangle(
            frame, (x_l, y_t), (x_r, y_b),
            (0, 255, 0), 2)

        # find y_pos to draw name, choose top if
        # there is space left
        # else: put it in bottom
        y = y_t - 15 if y_t > 30 else y_b + 15
        cv2.putText(frame, name, (x_l, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0), 2)

    # display fps
    t2 = time.time()
    fps = int(1 / (t2 - t1))
    fps_dis = "fps: {}".format(fps)
    cv2.putText(frame, fps_dis, (0, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255), 2)

    # ---- if set, write each frame to file ------------------
    # if video writer is None, and we are supposed to write
    # output video to disk, init the writer
    if writer is None and args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(
            args["output"], # name
            fourcc,         # 4cc
            20,             # fps
            (frame.shape[1], frame.shape[0]))   # size, (width, height)

    # if writer is not None, write frame to disk
    if writer is not None:
        writer.write(frame)

    # ---- if set, display each frame -------------------
    if args["display"] > 0:
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if `q` key was pressed, break
        if key == ord('q'):
            break

# Release everything if job is finished
print("[INFO] Release everything.")
cv2.destroyAllWindows()
video.stop()
if writer is not None:
    writer.release()
print("[INFO] Exit.")















































