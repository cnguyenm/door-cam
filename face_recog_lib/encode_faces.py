from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# global var
N_JITTERS = 5

# construct arg parser and parse the arg
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
                help="path to input dir of faces + images")
ap.add_argument("-o", "--output", required=True,
                help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str,
                default="hog",
                help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())


# grab the paths to the input images in our dataset
print("[SETUP] n_jitters = {}".format(N_JITTERS))
print("[INFO] loading image")
imagePaths = list(paths.list_images(args["dataset"]))

# init list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over image paths
# i =  1=> len(imagePaths)-1
# imagePath = "dataset/chau/file1.jpg"
for (i, imagePath) in enumerate(imagePaths):

    # extract name from image path
    # use i+1 because len(imagePath) count from 1
    print("[INFO] processing image {}/{}"
          .format(i+1, len(imagePaths)))

    # os.path.sep = '\'
    # get dir_name above image files
    # ex: dataset/chau/file1.png
    # output: chau
    name = imagePath.split(os.path.sep)[-2]

    # load input image, convert it BGR=>RBG
    # openCV: BGR, dlib: RGB
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect bboxes for each face
    boxes = face_recognition.face_locations(
        rgb,
        model=args["detection_method"]  # cnn, or hog
    )

    # compute facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes, num_jitters=N_JITTERS)

    # loop over encodings
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

# dump face encodings + names to disk
print("[INFO] serializing encodings ...")
data = {"encodings": knownEncodings,
        "names":knownNames}

f = open(args["output"], "wb")
pickle.dump(data, f)
f.close()
print("[INFO] done encoding")




























