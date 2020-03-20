import sys
import cv2
import glob
import os
from sys import platform
import argparse
import time
import numpy as np
import json

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
parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg",
                    help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../../models/"

# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1]) - 1:
        next_item = args[1][i + 1]
    else:
        next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-', '')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-', '')
        if key not in params: params[key] = next_item


if __name__ == '__main__':
    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Read frames on directory
    # imagePaths = op.get_images_on_directory(args[0].image_dir);
    walking_data_path = r'training_lstm/walking/'
    running_data_path = r'training_lstm/running/'
    other_data_path = r'training_lstm/other/'

    # list_path = (walking_data_path, running_data_path)
    # list_type = ('walking', 'running')
    list_path = [walking_data_path, running_data_path, other_data_path]
    list_type = ['walking', 'running', 'other']
    window = 20

for i in range(len(list_path)):
    data_path = list_path[i]
    file = glob.glob(os.path.join(data_path, '*.avi'))
    lstm_feature = np.zeros((0, window, 8))
    # index_train = np.random.default_rng().choice(len(file),int(round(len(file)/5*4)), replace=False)
    # index_test = np.delete(np.arange(len(file)),index_train)

    for index, video in enumerate(file):

        vidcap = cv2.VideoCapture(video)
        success, image = vidcap.read()
        t_start = time.time()
        t_list = []
        keypoints_list = []
        while success:
            datum = op.Datum()
            datum.cvInputData = image
            opWrapper.emplaceAndPop([datum])
            if len(datum.poseKeypoints.shape)==3:
                keypoints_list.append(datum.poseKeypoints[0, :, :])
                t_list.append(time.time() - t_start)
            success, image = vidcap.read()

        from people_feature_generation import Person
        keypoints_loc = Person().keypoints_loc
        keypoints_list = np.array(keypoints_list)
        t_list = np.array(t_list)
        valid_mask = np.all(~np.any(keypoints_list[:, keypoints_loc, 0:2] == 0, axis=2), axis=1)
        keypoints_list = keypoints_list[valid_mask,:,:]
        t_list = t_list[valid_mask]

        jump = [jumppoint for jumppoint in range(1,len(t_list)-2) if np.diff(t_list)[jumppoint] > 3*np.diff(t_list)[jumppoint-1] and np.diff(t_list)[jumppoint] > 3*np.diff(t_list)[jumppoint+1]]
        jump.insert(0,-1)
        jump.append(len(t_list))
        for jumppoint in range(len(jump)-1):
            # if not os.path.exists(os.path.join(os.getcwd(), 'training_lstm', list_type[i],'training_data')):
            #     os.mkdir(os.path.join(os.getcwd(), 'training_lstm',list_type[i],'training_data'))
            # json_filename = os.path.join(os.getcwd(), 'training_lstm', list_type[i], 'training_data', 'keypoint_%d_%d.json' % (index + 1, jumppoint+1))
            if jump[jumppoint+1] - jump[jumppoint] - 1 >= window:
                length = jump[jumppoint+1] - jump[jumppoint] - 1
                for k in range((length-window)//window):
                    valid_list = np.arange(jump[jumppoint]+1+window*k , jump[jumppoint]+1+window*(k+1))
                    data = keypoints_list[valid_list,:,:]
                    t = t_list[valid_list]
                    lstm_feature = np.concatenate((lstm_feature, Person().get_2d_angles(data,t)[:,0:8].reshape(1,window,8)), axis=0)
                if length % window != 0:
                    valid_list = np.arange(jump[jumppoint+1]-window,jump[jumppoint+1])
                    data = keypoints_list[valid_list, :, :]
                    t = t_list[valid_list]
                    lstm_feature = np.concatenate((lstm_feature, Person().get_2d_angles(data, t)[:, 0:8].reshape(1, window, 8)), axis=0)

    # filename = os.path.join(os.getcwd(), 'training_lstm', '%s.npy' % list_type[i])
    np.save('%s.npy' % list_type[i], lstm_feature)

            # data = {'body_keypoints': np.array(keypoints_list[jump[jumppoint]+1:jump[jumppoint+1]]).tolist(),
            #         'time_stamps': t_list[jump[jumppoint]+1:jump[jumppoint+1]]}
            # with open(json_filename, 'w') as write_file:
            #     json.dump(data, write_file)

        # json_filename = os.path.join('training_data',list_type[i],'keypoint_{%d}.json'%(index+1))
        # data = {'body_keypoints': np.array(keypoints_list).tolist(),
        #      'time_stamps': t_list}


