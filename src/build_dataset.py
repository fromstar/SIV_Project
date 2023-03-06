from camera import Camera
import cv2
import os

def build_data():
    url = 'http://192.168.1.147:8080/video'
    cam = Camera(url)

    for i in range(0,150):
        file_name = "dataset/my_data/" + str(i)+".png"
        img = cam.capture_frame()
        cv2.imwrite(file_name,img)

def fix_frames():
    path_p = os.listdir("dataset/pos/")
    path_n = os.listdir("dataset/neg/")
    
    for i in path_p:
        to_rw = "dataset/pos/"+i
        img = cv2.imread(to_rw)
        img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(to_rw,img)
    for i in path_n:
        to_rw = "dataset/neg/"+i
        img = cv2.imread(to_rw)
        img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(to_rw,img)

def new_dataset():
    negative_list = os.listdir("dataset/neg/")
    positive_list = os.listdir("dataset/pos/")
    for img in negative_list:
        frame = cv2.imread("dataset/neg/"+img)
        cv2.imwrite(("dataset/nn/n_"+img),frame)
    for img in positive_list:
        frame = cv2.imread("dataset/pos/"+img)
        cv2.imwrite(("dataset/pp/p_"+img),frame)

    negative_list = os.listdir("dataset/0/")
    positive_list = os.listdir("dataset/1/")
    for img in negative_list:
        frame = cv2.imread("dataset/0/"+img)
        cv2.imwrite(("dataset/nn/"+img),frame)
    for img in positive_list:
        frame = cv2.imread("dataset/1/"+img)
        cv2.imwrite(("dataset/pp/"+img),frame)

# build_data()
# fix_frames()
# new_dataset()