
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
print("[INFO] loading encodings")
data = pickle.loads(open(args["encodings"], "rb").read())

# init video stream and pointer to output video file
# VideoStream: wrapper class for cv2.Video
# diff: has a separate thread running
print("[INFO] start video stream")
video = VideoStream(src=0).start()
writer = None
time.sleep(2.0)

# loop over frames from video file stream
while True:
    # grab the frame from threaded video stream
    frame = video.read()

    # convert frame BGR<=>RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(rgb.shape[1])

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

    











































