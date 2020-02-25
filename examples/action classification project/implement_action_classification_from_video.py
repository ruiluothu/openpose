from skimage import feature

import sys
sys.path.append(r"C:\Users\Rui\Documents\GitHub\openpose\examples\tutorial_api_python")
import cv2
import os
from sys import platform
import argparse
import time
import json
import numpy as np
import glob
from classify_actions import get_action_classifier
from people_feature_generation import Person, ActionLabel, Joints
from scipy import interp

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))


def add_frame_overlay(img, probabilities):
    color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for i in range(len(probabilities)):
        if i == np.argmax(probabilities):
            color = color_list[i % len(color_list)]
        else:
            color = (255, 255, 255)
        cv2.putText(img, '{0:3.2f}'.format(probabilities[i]), (10, 30 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def _read_training_data(file_names):
    training_data = np.zeros([0, 25, 3])
    t_log = np.zeros([0])
    if isinstance(file_names, str):
        with open(file_names) as json_file:
            data = json.load(json_file)
            pos_data_arr = np.array(data['body_keypoints'])
            t_data_arr = np.array(data['time_stamps'])
            delta = np.diff(t_data_arr)
            jump = [index for index in range(1, delta.shape[0]-1) if delta[index]>3*delta[index+1] and delta[index]>3*delta[index-1]]
            jump.insert(0,-1)
            jump.append(t_data_arr.shape[0])
            if len(jump) !=2:
                print('no no no')

            for i in range(len(jump)-1):
                training_data = np.concatenate((training_data, pos_data_arr[jump[i]+1:jump[i+1],:,:]), axis=0)
                t_log = np.concatenate((t_log, t_data_arr[jump[i]+1:jump[i+1]]), axis=0)

    else:
        for file_name in file_names:
            with open(file_name) as json_file:
                data = json.load(json_file)
                pos_data_arr = np.array(data['body_keypoints'])
                t_data_arr = np.array(data['time_stamps'])
                training_data = np.concatenate((training_data, pos_data_arr), axis=0)
                t_log = np.concatenate((t_log, t_data_arr), axis=0)

    return training_data, t_log


if __name__ == '__main__':

    path1 = os.path.join('training', 'walking')
    path2 = os.path.join('training', 'running')
    files1 = glob.glob(os.path.join(path1, '*.json'))
    files2 = glob.glob(os.path.join(path2, '*.json'))

    # features_walk = np.zeros((0, 16))
    # features_run = np.zeros((0, 16))
    features_walk = np.zeros((0, 8))
    features_run = np.zeros((0, 8))

# Method1: get training data from all the files and concatenate them, the weak point is that during the combination of
# two videos, the feature of the interval will be bad

    # data, t = _read_training_data(files1)
    # features_walk = Person.get_2d_angles(data, t)
    # data, t = _read_training_data(files2)
    # features_run = Person.get_2d_angles(data, t)

# Method2: get data from each independent action sequence

    for file in files1:
        data, t = _read_training_data(file)
        features1 = Person.get_2d_angles(data, t)
        features_walk = np.concatenate((features_walk,features1),axis=0)

    for file in files2:
        data, t = _read_training_data(file)
        features2 = Person.get_2d_angles(data, t)
        features_run = np.concatenate((features_run, features2), axis=0)

    # in order to improve SVM performance, we normalize the feature data to [-1, 1]
    # no further need for grid search of hyper parameters
    features_walk = -1 + 2*(features_walk-np.min(features_walk, axis=0))/(np.max(features_walk, axis=0)-np.min(features_walk, axis=0))
    features_run = -1 + 2*(features_run-np.min(features_run, axis=0))/(np.max(features_run, axis=0)-np.min(features_run, axis=0))

    X = np.concatenate((features_walk,features_run), axis=0)
    y = np.concatenate((np.zeros(features_walk.shape[0]),np.ones(features_run.shape[0])),axis=0)


    from sklearn import svm
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.linear_model import LogisticRegression
    from classify_actions import lda

    clf1 = svm.SVC(kernel='rbf', gamma='scale')
    # clf2 = GaussianProcessClassifier(kernel=1.0 * RBF(1.0), random_state=0)
    # clf3 = LogisticRegression(max_iter=10000)
    # clf4 = lda()

    from sklearn.model_selection import cross_val_score
    scores1 = cross_val_score(clf1, X, y, cv=10)
    # scores2 = cross_val_score(clf2, X, y, cv=10)
    # scores3 = cross_val_score(clf3, X, y, cv=10)

    # train1 = np.random.default_rng().choice(X.shape[0],int(round(X.shape[0]/5*4)),replace=False)
    # test1 = np.delete(np.arange(X.shape[0]),train1)
    # clf4.fit(X[train1,:], y[train1])
    print(scores1)
    # print(scores1,scores2,scores3)
    # print(sum(clf4.predict(X[test1,:]) == y[test1])/test1.shape[0])








    # train1 = np.random.default_rng().choice(len(files1),int(round(len(files1)/5*4)),replace=False)
    # test1 = np.delete(np.arange(len(files1)),train1)
    # train2 = np.random.default_rng().choice(len(files2),int(round(len(files2)/5*4)),replace=False)
    # test2 = np.delete(np.arange(len(files2)), train2)
    #
    # truth = np.zeros(len(test1)+len(test2))
    # truth[0:len(test1)] = 0
    # truth[len(test1):len(test1)+len(test2)] = 1
    # print(truth)
    # accuracy_2d = np.zeros(len(test1) + len(test2))
    #
    # train1 = np.array(files1)[train1]
    # train2 = np.array(files2)[train2]
    # test1 = np.array(files1)[test1]
    # test2 = np.array(files2)[test2]
    # test = np.concatenate((test1,test2),axis=0)
    #
    # # train the classifier
    # action_classifier = get_action_classifier(False, None, [train1,train2])
    #
    # for index, jsonfile in enumerate(test):
    #     data, t = _read_training_data(jsonfile)
    #
    #     action_classifier = get_action_classifier(use_gpc=False, load_model_name='action_svm.pkl')
    #     label = Person.action_classification(data, t, action_classifier, use_probability=False)
    #
    #     accuracy_2d[index]=1-np.sum(label!=truth[index])/len(label)
    #     np.savetxt(os.path.join('test','label%d.csv' %(index+1)), label, delimiter=',')
    #
    # np.savetxt('accuracy_2d.csv',accuracy_2d,delimiter=',')


