"""
run with debug flag: -O
python -O abc.py

"""

import face_recognition
import argparse
import pickle
import cv2
import matplotlib.pyplot as plt
import time
import imutils

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
                help="path to serialized db of facial encodings")

ap.add_argument("-i", "--image", required=True,
                help="path to input image")

ap.add_argument("-d", "--detection", type=str, default="hog",
                help="detection model: `hog` or `cnn`")

args = vars(ap.parse_args())


# load known faces and embeddings
print("[INFO] loading encodings")
data = pickle.loads(open(args["encodings"], "rb").read())
print("[INFO] load {} encodings from db".format(len(data["encodings"])))

# load input image
t1 = time.time()
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# detect face location
print("[INFO] detect face locations")
boxes = face_recognition.face_locations(
    rgb,
    model=args["detection"]  # mostly 'hog', use `cnn` only if gpu
)

# if there is n_faces, output: n_encodings
encodings = face_recognition.face_encodings(rgb, boxes, num_jitters=2)

# initialize list of names for each face detected
names = []

# step:
# 1.foreach face found, compare that face with known db
# 2.then choose face that have most matches
# 3.if no match, leave name as Unknown
print("[INFO] recognizing face")
for e in encodings:
    # attempt to match each face in input image to our
    # known encodings
    # tolerance: default=0.6
    # return: list [T, F, T, ...]
    matches = face_recognition.compare_faces(
        data["encodings"], e
    )
    name = "Unknown"

    # check if we have a match
    if True in matches:
        # find indexes of al matched faces, then
        # init a dict to count the total number of times
        # each face was matched
        matchedIds = [i for (i,b) in enumerate(matches) if b]
        counts = {}

        # loop over the matched indexes
        # maintain a count for each recognized face
        for i in matchedIds:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1

        # determine the recognized face with
        # largest number of votes
        # (if tie, Python will select 1st entry)
        name = max(counts, key=counts.get)

    # update list of names
    names.append(name)


# now, draw the recognize face
# because boxes => encodings => names
# so index should be the same
print("[INFO] draw face rect")
for ((y_t, x_r, y_b, x_l), name) in zip(boxes, names):

    # draw face rect
    cv2.rectangle(image, (x_l, y_t), (x_r, y_b), (0, 255, 0), 2)

    # find y_pos to draw name, choose top if
    # there is space left
    # else: put it in bottom
    y = y_t - 15 if y_t > 30 else y_b + 15
    cv2.putText(image, name, (x_l, y), cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0), 2)

# show output image
t2 = time.time()
print("[INFO] Time: " + str(t2 - t1))
print("[INFO] display result")
print("[INFO] Names: " + str(names))
output_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(output_image)
plt.show()
print("[INFO] Exit.")

































