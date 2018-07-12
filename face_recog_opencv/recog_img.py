"""
Good file to test the accuracy of training data

Test predcit image
"""

import cv2
import pickle
import argparse

# global
FACE_SIZE = 165
subjects = ["dummy label", "Khoa", "C Tu", "Chau"]
face_recognizer = None 
name_list = []


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
    if text is None:
        name = "Unknown"
    else:
        name = text 

    cv2.putText(
        img, name, (x,y-8), cv2.FONT_HERSHEY_PLAIN, 1.5, 
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
    face = cv2.resize(gray[y: y+h, x:x+w], (FACE_SIZE, FACE_SIZE))
    return face, faces[0]

def predict(original_img):
    """
    given an opencv image, predict name
    """

    # copy, don't want to change original img
    img = original_img.copy()

    # detect face
    face, rect = detect_face(img)
    if face is None:
        print("no face detected")
        return original_img

    # predict face, get label or id
    label, confidence = face_recognizer.predict(face)

    # get name
    # subjects should be generated when training data, or classifer
    # because not guarantee to detect face in every image
    # for the sake of testing, this is dummy data
    name = name_list[label]  

    # draw face rect
    draw_rect(img, rect, name + ": " + str(int(confidence)) + "%")

    # return img with rect
    return img

def init(name_file, db_file):
    
    global name_list
    global face_recognizer

    # load name_list
    name_list = None
    try:
        with open(name_file, 'rb') as fp:
            name_list = pickle.load(fp) 
    except FileNotFoundError:
        print("[ERROR] cannot read file_name")
        exit(0)

    # load face_recognizer
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    try:
        face_recognizer.read(db_file)
    except cv2.error as e:
        print("[ERROR] cannot read file_db")
        exit(0)

def main():

    # parse arg
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--database", required=True,
                help="training data, in yml")
    ap.add_argument("-n", "--name", required=True,
                help="file name, in pickle")
    ap.add_argument("-i", "--image", required=True,
                help="input image")

    args = vars(ap.parse_args())

    # global
    global face_recognizer
    global name_list

    # load namelist
    # name_list = None
    # try:
    #     with open(args['name'], 'rb') as fp:
    #         name_list = pickle.load(fp) 
    # except FileNotFoundError:
    #     print("[ERROR] cannot read file_name")
    #     exit(0)
    init(args['name'], args['database'])

    # create LBPH face recognizer
    #face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    

    #print(name_list)

    # load
    test_img = cv2.imread(args['image'])
    if test_img is None:
        print("Cannot read image")
        exit(0)

    # resize
    h, w = test_img.shape[:2]
    w2 = 600
    r  = w2/w 
    h2 = int(h * r)
    img = cv2.resize(test_img, (w2, h2))

    # predict
    predict_img = predict(img)

    # show result
    cv2.imshow("result", predict_img)
    cv2.waitKey(0) & 0xff
    cv2.destroyAllWindows()


if __name__=='__main__':
    main()