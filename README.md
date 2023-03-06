# SIV_Project

In this project a Motion Detection + Human Identification is implemented. 
The project focuses on the implementation of the Histogram of Oriented Gradients(HOG) method for the part of the Human identification.
The motion detection is done through a Background Substraction method while the Classification part is done using a linear SVM fed with the HOG descriptor.

## Requirements
* Python V3.10
* Cv2 V4.7.0
* Numpy V1.24.2
* Skimage V0.19.3
* Sklearn V1.2.1
* Joblib V1.2.0
* Tqdm V4.65.0
* Matplotlib V3.7.0

## Dataset used
A mixture of two dataset is used. The positive samples are taken from the Market-1501 Attribute dataset while the negative samples are taken from the the VIRAT 2.0 dataset. In particular the Virat frames are taken from this repository: https://github.com/agikarasugi/HumanBinaryClassificationSuite.git.

## Run the code
To run the code just open the main.py file and change the url with the ip of your ip-camera. If you don't have an ip camera and you want to test with the webcam of your computer just substitute the url with an integer 0 value.

## Train the model
If you want to train the model and the run the code set to True the "t" flag in the main file.
If you just want to train the model without run the main code go to the SVM.py file and set to True the "t" flag. 
The model can be trained and tested with the precomputed hogs descriptor but it is possibile recompute them with different parameters. Just set to False the "train_load" and "test_load". The parameters to modify for the hog descriptors computing are in the hog.py file.
