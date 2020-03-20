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
import joblib
from people_feature_generation import Person
from sklearn import svm


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


def _reading_json_file(files):
    training_data = np.zeros((0,25,3))
    training_time = np.zeros(0)
    if isinstance(files, str):
        with open(files) as json_file:
            json_file = json.load(json_file)
            data = np.array(json_file['body_keypoints'])
            t = np.array(json_file['time_stamps'])
            delta = np.diff(t)
            jumppoint = [jump for jump in range(1,delta.shape[0]-1) if delta[jump]>3*delta[jump+1] and delta[jump]>3*delta[jump-1]]
            jumppoint.insert(0,-1)
            jumppoint.append(t.shape[0])
            for i in range(len(jumppoint)-1):
                pos_data = data[jumppoint[i]+1:jumppoint[i+1]]
                time_data = t[jumppoint[i]+1:jumppoint[i+1]]
                training_data = np.concatenate((training_data, pos_data), axis=0)
                training_time = np.concatenate((training_time, time_data), axis=0)

    else:
        print('Reading multiple files... ')
        print('Might cause problems')
        for file in files:
            with open(file) as json_file:
                json_file = json.load(json_file)
                data = np.array(json_file['body_keypoints'])
                t = np.array(json_file['time_stamps'])
                training_data = np.concatenate((training_data, data), axis=0)
                training_time = np.concatenate((training_time, t), axis=0)

    return training_data, training_time


if __name__ == '__main__':

    _default_save_model_name = 'svm_all_features.pkl'

    if not os.path.exists(_default_save_model_name):
        walking_data_path = r'training/walking/'
        running_data_path = r'training/running/'
        other_data_path = r'training/other/'

        list_path = [walking_data_path, running_data_path, other_data_path]
        list_type = ['walking', 'running', 'other']


        X = np.zeros((0,24))
        y = np.zeros(0)
        for i in range(len(list_path)):
            path = list_path[i]
            label = np.array([i])
            for file in glob.glob(os.path.join(path, '*.json')):
                data, t = _reading_json_file(file)
                feature = Person().get_2d_angles(data, t)
                X = np.concatenate((X, feature), axis=0)
                y = np.concatenate((y, label*np.ones(feature.shape[0])), axis=0)

        clf = svm.SVC(kernel='rbf', gamma='scale')
        clf.fit(X, y)
        save_model_name = _default_save_model_name
        joblib.dump(clf, save_model_name)

    else:
        model = joblib.load(_default_save_model_name)

