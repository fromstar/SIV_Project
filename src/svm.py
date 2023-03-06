from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

import joblib
from hog import pre_built_hog
from hog import compute_hog

import cv2
import os
from tqdm import tqdm

neg_dir = "dataset/0/"
pos_dir = "dataset/1/"
n_val = 40

def prepare_dataset():
    negative_list = os.listdir(neg_dir)
    positive_list = os.listdir(pos_dir)

    negative_list = negative_list[n_val:]
    positive_list = positive_list[n_val:]

    pbar = tqdm(total=len(negative_list)+len(positive_list))

    labels = []
    hogs = []
    for img in negative_list:
        frame = cv2.imread(neg_dir+img)
        # hogs.append(pre_built_hog(frame))
        hogs.append(compute_hog(frame))
        labels.append(0)
        pbar.update(1)

    for img in positive_list:
        frame = cv2.imread(pos_dir+img)
        # hogs.append(pre_built_hog(frame))
        hogs.append(compute_hog(frame))
        labels.append(1)
        pbar.update(1)

    joblib.dump(hogs,"models/train_hogs.pkl")
    joblib.dump(labels,"models/train_labels.pkl")

def test(load):
    clf = joblib.load("models/model.pkl")

    total_cnt = 0
    positive_cnt = 0
    if load == True:
        hogs = joblib.load("models/val_hogs.pkl")
        labels = joblib.load("models/val_labels.pkl")
        for hog,label in zip(hogs,labels):
            predict = clf.predict([hog])
            if predict[0] == label:
                positive_cnt +=1
            total_cnt +=1
    else:
        negative_list = os.listdir(neg_dir)
        positive_list = os.listdir(pos_dir)

        negative_list = negative_list[0:n_val]
        positive_list = positive_list[0:n_val]

        hogs = []
        labels = []

        pbar = tqdm(total=len(negative_list)+len(positive_list))

        for img in negative_list:
            frame = cv2.imread(neg_dir+img)
            # hog = pre_built_hog(frame)
            hogs = joblib.load("models/train_hogs.pkl")
            hog = compute_hog(frame)
            predict = clf.predict([hog])
            total_cnt += 1
            if predict[0] == 0:
                positive_cnt += 1
            hogs.append(hog)
            labels.append(0)
            pbar.update(1)

        for img in positive_list:
            frame = cv2.imread(pos_dir+img)
            # hog = pre_built_hog(frame)
            hog = compute_hog(frame)
            predict = clf.predict([hog])
            total_cnt += 1
            if predict[0] == 1:
                positive_cnt += 1
            hogs.append(hog)
            labels.append(1)
            pbar.update(1)
        
        joblib.dump(hogs,"models/val_hogs.pkl")
        joblib.dump(labels,"models/val_labels.pkl")
    
    print("Accuracy: " + str((positive_cnt/total_cnt)*100) + "%")

def train(load):
    if load == False:
        prepare_dataset()
    
    hogs_train = joblib.load("models/train_hogs.pkl")
    labels_train = joblib.load("models/train_labels.pkl")
    clf = svm.SVC(kernel='linear')
    
    clf.fit(hogs_train,labels_train)
    joblib.dump(clf,"models/model.pkl")

train(load = False)
test(load = False)