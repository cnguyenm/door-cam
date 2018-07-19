# Door_cam
[![Build status](https://img.shields.io/badge/Build-TBD-red.svg)](#)
[![Developing](https://img.shields.io/badge/Developing-On%20Progress-brightgreen.svg)](#)

Door_cam is a software to put in the front-door camera to recognize people and track their number of times entering. So we can collect data or simply know who come in regularly. 
Right now, it can 
* detect & track face, 
* recognize that face using 
	* face_recognition. [Github face_recognition](https://github.com/ageitgey/face_recognition)
	* or python openCV EigenFace

For now, it works fine, with good performance (about 30fps), though the accuracy is not there.
It would be better with more training images for EigenFaceRecognizer
![Demo](/img/demo_doorcam2.png)



# Installation
I develop on Linux Ubuntu 16.04 LTS, but should work well with Windows
```
git clone https://github.com/cnguyenm/door_cam.git
cd door_cam
pip install -r requirements.txt
```
Remember to have a dataset of people faces. Its structure should be like:
```
-name1
	-- p1-face.png
	-- p2-face2.jpg
	-- face3.png
-name2
	-- p2-face.jpg
	-- p2325.jpg
```
Name of picture file is not important. It would be good have as many pictures as you can. (better if frontal face)

## Train and run with face_recognition lib
### Train
If you only have CPU, then you can use detection method `hog` to detect face
```
cd face_recog_lib/
python3 encode_faces.py --dataset <path_dataset> --output <path_output>
```

if you have GPU and CUDA enabled, you can use method `cnn`
```
python3 encode_faces.py --dataset <path_dataset> --output <path_output> -d cnn
``` 
output: a file.pickle with face encodings

### Run
Run with input 0 for webcam, or your video_path
```
python3 door_cam.py -e <path_encodings> -i 0
```
And if there is GPU & CUDA, use cnn method for better face detection accuracy
```
python3 door_cam.py -e <path_encodings> -i 0 -d cnn
```


## Train and run with openCV EigenFace lib

A built-in face recognizer of openCV. You should have library opencv-contrib-python installed, instead of opencv-python. Because of some copy-right law, opencv-python has removed some of its packages.

### Train
```
cd face_recog_lib
python3 train.py --dataset <dataset>
```
This output 
* a db.yml file for training
* a name.pickle file for name of people
You can rename them.

### Run
```
python3 door_cam2.py -d <dataset> -n <name_file>
```
