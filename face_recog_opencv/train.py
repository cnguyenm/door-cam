
import cv2
import os
import datetime
import pickle
import argparse
import numpy as np
from imutils import paths

# global var
debug = True 
FACE_SIZE = 165



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

def prepare_training_data(dataset_path):
    """
    prepare data for training

    :param dataset_path, [str], path to dataset
    :return face_list, [list<opencv images>], contain cropped images
    :return id_list, [list<int>], contains ids correspond to face
    :return name_list, [list<str>], list name, should be once for a name
    because id will be the index of name_list

    """

    # get all image paths
    # list: [path1, path2, ...]
    imagePaths = list(paths.list_images(dataset_path))

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

def test1():
    img = cv2.imread("../img/test_chau2.jpg")
    face, rect = detect_face(img)

    cv2.imshow("face", face)
    k = cv2.waitKey(0) 
    cv2.destroyAllWindows()


def main():
    
    # parse arg
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                help="path to dataset images")
    args = vars(ap.parse_args())

    # prepare data
    debug("[INFO] prepare data...")
    face_list, id_list, name_list = prepare_training_data(args["dataset"])
    
    # print
    debug("[INFO] Total faces: " + str(len(face_list)))
    debug("[INFO] Total labels: " + str(len(id_list)))    

    # train
    # create LBPH face recognizer
    debug("[INFO] training data: EigenFace")
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    #face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(face_list, np.array(id_list))

    # create a unique name to save file
    # because I don't want to many cmd args
    # now: type = str
    now = datetime.datetime.now().strftime('%Y-%m-%d@%H:%M:%S') 

    # save training data
    db_path = "db{}.yml".format(now)
    debug("[INFO] output: train file: " + db_path)
    face_recognizer.save(db_path)

    # save name list
    name_path = "name{}.pickle".format(now)
    debug("[INFO] output: name file: " + name_path)
    file_handler = open(name_path, 'wb')
    pickle.dump(name_list, file_handler)
    file_handler.close()

    # exit
    debug("[INFO] You can change file_names later. :-| ")
    debug("[INFO] Exit.")



if __name__=='__main__':
    main()