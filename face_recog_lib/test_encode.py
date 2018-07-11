
import cv2
import face_recognition

imagePath = "dataset/chau/chau3.jpg"


image = cv2.imread(imagePath)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detect bboxes for each face
print("detect face...")
boxes = face_recognition.face_locations(rgb, model="hog")
print(boxes)

# compute facial embedding for the face
print("encode face")

# if n faces, n encodings
# assume all faces belong to same person
encodings = face_recognition.face_encodings(rgb, boxes)
print(len(encodings))