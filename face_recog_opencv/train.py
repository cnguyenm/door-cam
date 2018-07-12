
import cv2
import os
import numpy as np
from imutils import paths

# global var
debug = True 
subjects = ["dummy label", "Khoa", "C Tu", "Chau"]
FACE_SIZE = 165

# create LBPH face recognizer
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

def debug(msg, end='\n'):
    """
    useful functions because I don't want to write 
    long code
    """
    print(msg, end=end) if debug else 0

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

def detect_face(img):
    """
    function to detect face, using openCV

    :param img, <ndarray> input image
    """

    # convert img to gray scale, as opencv face detector expects
    # gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # load face detector
    face_cascade = cv2.CascadeClassifier("../cv_train/haarcascade_frontalface_default.xml")

    # detect face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    # if no faces detected, return orignal img
    if len(faces) == 0:
        return None, None 

    # extract first face
    (x, y, w, h) = faces[0]
    
    # return only face part of image
    # w == h. so I guess it's fine to use
    # either to slice
    return gray[y: y+w, x:x+h], faces[0]

def detect_face2(img):
    """
    function to detect face, using openCV
    different from detect_face: this will 
    return same size for all faces. 
    EigenFace, FisherFace requires same size

    :param img, <ndarray> input image
    """

    # convert img to gray scale, as opencv face detector expects
    # gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # load face detector
    face_cascade = cv2.CascadeClassifier("../cv_train/haarcascade_frontalface_default.xml")

    # detect face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    # if no faces detected, return orignal img
    if len(faces) == 0:
        return None, None 

    # extract first face
    (x, y, w, h) = faces[0]
    
    # return only face part of image
    # w == h. so I guess it's fine to use
    # either to slice
    face = gray[y:y+w, x:x+h]
    face = cv2.resize(face, (FACE_SIZE, FACE_SIZE))

    return face, faces[0]

def prepare_training_data():
    """
    prepare data for training

    :return face_list, list<opencv images>, contain cropped images
    :return id_list, list<int>, contains ids correspond to face
    :return name_list, list<str>, list name, should be once for a name
    because id will be the index of name_list

    """

    # get all image paths
    # list: [path1, path2, ...]
    imagePaths = list(paths.list_images("../dataset_mini"))

    # var
    old_name = None 
    name = None 
    id = 0
    id_list = []
    face_list = []
    name_list = [""]    # first name is dummy

    # loop over image paths
    # i =  1=> len(imagePaths)-1
    # imagePath = "dataset/chau/file1.jpg"
    for (i, imagePath) in enumerate(imagePaths):
    
        # os.path.sep = '\'
        # get dir_name above image files
        # ex: dataset/chau/file1.png
        # output: chau
        name = imagePath.split(os.path.sep)[-2]

        # detect face
        debug("[INFO] detect face: {}/{}: {}".format(i+1, len(imagePaths), name), end=" ")
        image = cv2.imread(imagePath)
        gray_face, rect = detect_face2(image)

        # if face not found
        if gray_face is None:
            debug("=> face not found")
            continue

        # if face found
        # if change person, update name
        # update label
        if old_name != name:
            id += 1
            old_name = name
            name_list.append(name)
        
        # add to list
        debug("=> OK")
        id_list.append(id)
        face_list.append(gray_face)
        

    return face_list, id_list, name_list

def predict(original_img):
    """
    given an opencv image, predict name
    """

    # copy, don't want to change original img
    img = original_img.copy()

    # detect face
    face, rect = detect_face(img)

    # predict face, get label or id
    label, confidence = face_recognizer.predict(face)

    # get name
    # subjects should be generated when training data, or classifer
    # because not guarantee to detect face in every image
    # for the sake of testing, this is dummy data
    name = subjects[label]  

    # draw face rect
    draw_rect(img, rect, name + ": " + str(int(confidence) + "%"))

    # return img with rect
    return img


def main():
    
    # prepare data
    debug("[INFO] prepare data...")
    face_list, id_list, name_list = prepare_training_data()
    
    # print
    debug("[INFO] Total faces: " + str(len(face_list)))
    debug("[INFO] Total labels: " + str(len(id_list)))
    debug("[DEBUG] " + str(id_list))    

    # train
    debug("[INFO] training data...")
    #face_recognizer.train(face_list, np.array(id_list))

    # # save data
    # debug("[INFO] saving data in db2.yml")
    # face_recognizer.save("db2.yml")

def test1():
    img = cv2.imread("../img/test_chau2.jpg")
    face, rect = detect_face(img)

    cv2.imshow("face", face)
    k = cv2.waitKey(0) 
    cv2.destroyAllWindows()

main()