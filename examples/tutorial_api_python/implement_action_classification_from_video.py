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
from classify_actions import get_action_classifier_all, get_action_classifier_2d_only, get_action_classifier_3d_only
from people_feature_generation import Person, ActionLabel, Joints
from scipy import interp

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/../../build/python/openpose/Release');
        os.environ['PATH'] = os.environ[
                                 'PATH'] + ';' + dir_path + '/../../build/x64/Release;' + dir_path + '/../../build/bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('../../python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", default="../../../examples/media/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
parser.add_argument("--net_resolution", default="1x176")
parser.add_argument("--disable_multi_thread", default=True)
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../models/"
# params['render_threshold'] = 0.15

# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1])-1: next_item = args[1][i+1]
    else: next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-','')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-','')
        if key not in params: params[key] = next_item


def add_frame_overlay(img, probabilities):
    color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for i in range(len(probabilities)):
        if i == np.argmax(probabilities):
            color = color_list[i % len(color_list)]
        else:
            color = (255, 255, 255)
        cv2.putText(img, '{0:3.2f}'.format(probabilities[i]), (10, 30 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


if __name__ == '__main__':
    # start OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    test_path = 'test'
    test = glob.glob(os.path.join(test_path, '*.avi'))
    truth = np.zeros(len(test))
    truth[np.array([i for i, val in enumerate(test) if 'walking' in val])] = 0
    # truth[np.array([i for i, val in enumerate(test) if 'jogging' in val])] = 1
    truth[np.array([i for i, val in enumerate(test) if 'running' in val])] = 1
    print(truth)
    accuracy_all = np.zeros(len(test))
    accuracy_2d = np.zeros(len(test))
    accuracy_3d = np.zeros(len(test))

    # train the classifier
    # action_classifier1 = get_action_classifier_all(use_gpc=False, load_model_name=None)
    # action_classifier2 = get_action_classifier_2d_only(use_gpc=False, load_model_name=None)
    # action_classifier3 = get_action_classifier_3d_only(use_gpc=False, load_model_name=None)

    for index, video in enumerate(test):
        # first turn the test videos to a json file with keypoint and time info
        vidcap = cv2.VideoCapture(video)
        height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        success, image = vidcap.read()
        # count = 0
        t_start = time.time()
        t_list = np.zeros(0)
        keypoints_list = np.zeros((0,25,3))
        while success:
            # img_loc = "frame%d.jpg" % count
            # cv2.imwrite(os.path.join(walking_data_path, "%d" % (index+1), "%d frame%d.jpg" % (index+1, count)), image)  # save frame as JPEG file
            datum = op.Datum()
            datum.cvInputData = image
            opWrapper.emplaceAndPop([datum])
            if len(datum.poseKeypoints.shape) == 3:
                keypoints_list=np.concatenate((keypoints_list, datum.poseKeypoints[0,:,:].reshape(1,25,3)), axis=0)
                t_list=np.concatenate((t_list, np.array([time.time() - t_start])), axis=0)
                # if t_list.shape[0] != keypoints_list.shape[0]:
                #     print(t_list.shape,keypoints_list.shape)
            success, image = vidcap.read()
            # count += 1
# 1
        action_classifier = get_action_classifier_all(use_gpc=False, load_model_name='action_svm_all.pkl')
        # person = Person()
        # if ~np.any(keypoints_list[:, person.keypoints_loc, :] == 0, axis=2):
        if t_list.shape[0] != keypoints_list.shape[0]:
            t_list=np.concatenate((t_list,interp(np.arange(t_list.shape[0],keypoints_list.shape[0]),np.arange(t_list.shape[0]),t_list)),axis=0)

        label = Person.action_classification_all(keypoints_list, t_list, action_classifier, use_probability=True)


        accuracy_all[index]=1-np.sum(label!=truth[index])/len(label)
        np.savetxt(os.path.join(test_path,'all','label%d.csv' %(index+1)), label, delimiter=',')


# 2
        action_classifier = get_action_classifier_2d_only(use_gpc=False, load_model_name='action_svm_2d.pkl')
        # person = Person()
        # if ~np.any(keypoints_list[:, person.keypoints_loc, :] == 0, axis=2):
        if t_list.shape[0] != keypoints_list.shape[0]:
            t_list=np.concatenate((t_list,interp(np.arange(t_list.shape[0],keypoints_list.shape[0]),np.arange(t_list.shape[0]),t_list)),axis=0)

        label = Person.action_classification_2d(keypoints_list, t_list, action_classifier, use_probability=True)


        accuracy_2d[index]=1-np.sum(label!=truth[index])/len(label)
        np.savetxt(os.path.join(test_path,'2d','label%d.csv' %(index+1)), label, delimiter=',')


# 3
        action_classifier = get_action_classifier_3d_only(use_gpc=False, load_model_name='action_svm_3d.pkl')
        # person = Person()
        # if ~np.any(keypoints_list[:, person.keypoints_loc, :] == 0, axis=2):
        if t_list.shape[0] != keypoints_list.shape[0]:
            t_list=np.concatenate((t_list,interp(np.arange(t_list.shape[0],keypoints_list.shape[0]),np.arange(t_list.shape[0]),t_list)),axis=0)

        label = Person.action_classification_3d(keypoints_list, t_list, action_classifier, use_probability=True)


        accuracy_3d[index]=1-np.sum(label!=truth[index])/len(label)
        np.savetxt(os.path.join(test_path,'3d','label%d.csv' %(index+1)), label, delimiter=',')

    np.savetxt('accuracy_all.csv',accuracy_all,delimiter=',')
    np.savetxt('accuracy_2d.csv', accuracy_2d, delimiter=',')
    np.savetxt('accuracy_3d.csv', accuracy_3d, delimiter=',')


def _read_training_data(file_names):
    training_data = np.zeros([0, 25, 3])
    t_log = np.zeros([0])
    for file_name in file_names:
        with open(file_name) as json_file:
            data = json.load(json_file)
            pos_data_arr = np.array(data['body_keypoints'])
            t_data_arr = np.array(data['time_stamps'])
            training_data = np.concatenate((training_data, pos_data_arr), axis=0)
            t_log = np.concatenate((t_log, t_data_arr), axis=0)

    return training_data, t_log