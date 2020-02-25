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
    truth[np.array([i for i, val in enumerate(test) if 'jogging' in val])] = 1
    truth[np.array([i for i, val in enumerate(test) if 'running' in val])] = 2
    print(truth)
    accuracy = np.zeros(len(test))


    # sys.path.append('../../python');
    action_classifier = get_action_classifier(use_gpc=False, load_model_name=None)

    for index, video in enumerate(test):
        dir = os.path.join(test_path, "%d" % (index+1))
        if not os.path.exists(dir):
            os.mkdir(dir)

        vidcap = cv2.VideoCapture(video)
        height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)

        date_time_now = time.strftime('%Y-%m-%d-%H%M%S', time.localtime())
        # video_out_file = os.path.join(dir,'video_{0}.avi'.format(date_time_now))
        # video_writer = cv2.VideoWriter(video_out_file, cv2.VideoWriter_fourcc(*'XVID'), 20, (int(width), int(height)))

        success, image = vidcap.read()
        action_classifier = get_action_classifier(use_gpc=False, load_model_name= 'action_svm.pkl')
        t_stamp = np.repeat(time.time(),3)
        # t_list = []
        # keypoints_list = []
        label_list = np.zeros(0)
        keypoint_list = np.zeros((0,25,3))

        while success and keypoint_list.shape[0] != 3:
            # cv2.imwrite(os.path.join(walking_data_path, "%d" % (index+1), "%d frame%d.jpg" % (index+1, count)), image)  # save frame as JPEG file
            datum = op.Datum()
            datum.cvInputData = image
            opWrapper.emplaceAndPop([datum])
            if len(datum.poseKeypoints.shape)==3:
                keypoint_list = np.concatenate((keypoint_list,datum.poseKeypoints),axis=0)
                t_stamp = np.hstack((t_stamp[1:3], time.time()))

            # video_writer.write(datum.cvOutputData)
            success, image = vidcap.read()

        while success:
            # cv2.imwrite(os.path.join(walking_data_path, "%d" % (index+1), "%d frame%d.jpg" % (index+1, count)), image)  # save frame as JPEG file
            datum = op.Datum()
            datum.cvInputData = image
            opWrapper.emplaceAndPop([datum])
            if len(datum.poseKeypoints.shape)==3:
                keypoint_list = np.concatenate((keypoint_list[1:3, :, :], datum.poseKeypoints), axis=0)
                person = Person()
                if ~np.any(keypoint_list[:, person.keypoints_loc, :] == 0):
                    t_stamp = np.hstack((t_stamp[1:3], time.time()))
                    label = Person.update_action_classification(keypoint_list, t_stamp, action_classifier,
                                                            use_probability=False)
                    label_list = np.concatenate((label_list,np.array(label.value).flatten()),axis=0)

                # keypoints_list.append(datum.poseKeypoints[0, :, :])
                # t_list.append(time.time() - t_start)
                # datum.poseKeypoints.reshape[]
                # Person.update_pose_keypoints(pose_keypoints[0, :, :], t_stamp=time.time())
                # print(datum.poseKeypoints[0, :, :])
                # t_stamp = np.hstack((t_stamp[1:3],time.time()))
                # label = Person.update_action_classification(datum.poseKeypoints[0, :, :], t_stamp, action_classifier, use_probability=False)
                # label_list = [label_list, label]
                # probabilities = person.get_filtered_label_probabilities()
                # add_frame_overlay(img, probabilities)
                # fps = 1. / (time.time() - t_start)
                # cv2.putText(img, '{0:3.1f}'.format(fps), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                #             (255, 255, 255), 2)

                # if label is ActionLabel.pointing:
                # alpha = Person.get_rh_angle_with_image_plane(pose_keypoints[0, :, :], focal_length)
                # if alpha is not None:
                #     v_x_command = 50 * np.sin(alpha)
                #     v_y_command = 50 * np.cos(alpha)
                #     pt_1 = (60, 160)
                #     cv2.line(img,
                #              pt_1,
                #              (int(pt_1[0] - v_y_command), int(pt_1[1] + v_x_command)),
                #              (255, 255, 255),
                #              4)

            video_writer.write(datum.cvOutputData)
            success, image = vidcap.read()

        accuracy[index]=1-np.sum(label_list!=truth[index])/len(label_list)
        np.savetxt(os.path.join(test_path,'label%d.csv' %(index+1)), label_list, delimiter=',')

    np.savetxt('accuracy.csv',accuracy,delimiter=',')
        # data = {'classify_label': label_list}
        # json_filename = os.path.join(test_path, 'label_{0}.json'.format(date_time_now))
        # with open(json_filename, 'w') as write_file:
        #     json.dump(data, write_file)

            # person.update_pose_keypoints(pose_keypoints[0, :, :], t_stamp=time.time())
            # label = person.update_action_classification(action_classifier, use_probability=True)
            # probabilities = person.get_filtered_label_probabilities()
            # add_frame_overlay(img, probabilities)
            # fps = 1. / (time.time() - t_prev)
            # cv2.putText(img, '{0:3.1f}'.format(fps), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #             (255, 255, 255), 2)
            #
            # # if label is ActionLabel.pointing:
            # alpha = Person.get_rh_angle_with_image_plane(pose_keypoints[0, :, :], focal_length)
            # if alpha is not None:
            #     v_x_command = 50 * np.sin(alpha)
            #     v_y_command = 50 * np.cos(alpha)
            #     pt_1 = (60, 160)
            #     cv2.line(img,
            #              pt_1,
            #              (int(pt_1[0] - v_y_command), int(pt_1[1] + v_x_command)),
            #              (255, 255, 255),
            #              4)











    # while cap.isOpened():
    #     datum = op.Datum()
    #
    #     ret, frame = cap.read()
    #
    #     if ret:
    #         datum.cvInputData = frame
    #         opWrapper.emplaceAndPop([datum])
    #
    #         img = datum.cvOutputData
    #         pose_keypoints = datum.poseKeypoints
    #         if np.size(pose_keypoints) > 1:
    #             person.update_pose_keypoints(pose_keypoints[0, :, :], t_stamp=time.time())
    #             label = person.update_action_classification(action_classifier, use_probability=True)
    #             probabilities = person.get_filtered_label_probabilities()
    #             add_frame_overlay(img, probabilities)
    #             fps = 1. / (time.time() - t_prev)
    #             cv2.putText(img, '{0:3.1f}'.format(fps), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #                         (255, 255, 255), 2)
    #
    #             # if label is ActionLabel.pointing:
    #             alpha = Person.get_rh_angle_with_image_plane(pose_keypoints[0, :, :], focal_length)
    #             if alpha is not None:
    #                 v_x_command = 50 * np.sin(alpha)
    #                 v_y_command = 50 * np.cos(alpha)
    #                 pt_1 = (60, 160)
    #                 cv2.line(img,
    #                          pt_1,
    #                          (int(pt_1[0] - v_y_command), int(pt_1[1] + v_x_command)),
    #                          (255, 255, 255),
    #                          4)
    #         else:
    #             print('nobody detected')
    #
    #         cv2.imshow('OpenPose', img)
    #         t_prev = time.time()
    #         key = cv2.waitKey(1)
    #         if key == 27:
    #             break
    #
    # cap.release()



