
import cv2


FACE_SIZE = 165
subjects = ["dummy label", "Khoa", "C Tu", "Chau"]

s2 = [
    "dumb", "a viet", "a khoa", "t hung",
    "linh", "tung", "T ha", "c ngoc anh", "a TTduc",
    "c tu", "a Thang", "a Dat", "T Thang", "Huy", "A Giang",
    "Xuan", "a Thanh", "A Dung", "a Duc", "T Tu",
    "A Hung", "C Thanh", "Chau", "Lien", "A Bach"
]

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
    name = s2[label]  

    # draw face rect
    draw_rect(img, rect, name + ": " + str(int(confidence)) + "%")

    # return img with rect
    return img


# create LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer.read("db2.yml")

# load
test_img = cv2.imread("../img/test_chau2.jpg")

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