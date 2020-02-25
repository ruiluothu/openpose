import sys
import cv2
import glob
import os
from sys import platform
import argparse
import time
import numpy as np
import json
import warnings
# warnings.simplefilter('always', UserWarning)

# try:
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
    walking_data_path = r'training_data/walking/'
    jogging_data_path = r'training_data/jogging/'
    running_data_path = r'training_data/running/'

    list_path = (walking_data_path,jogging_data_path,running_data_path)
    list_type = ('walking','jogging','running')
for i in range(len(list_path)):
    data_path = list_path[i]
    walking = glob.glob(os.path.join(data_path, '*.avi'))
    # sys.path.append('../../python');

    for index, video in enumerate(walking):
        dir = os.path.join(data_path, "%d" % (index+1))
        if not os.path.exists(dir):
            os.mkdir(dir)

        vidcap = cv2.VideoCapture(video)
        height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)

        date_time_now = time.strftime('%Y-%m-%d-%H%M%S', time.localtime())
        video_out_file = os.path.join(dir,'video_{0}.avi'.format(date_time_now))
        video_writer = cv2.VideoWriter(video_out_file, cv2.VideoWriter_fourcc(*'XVID'), 20, (int(width), int(height)))

        success, image = vidcap.read()
        count = 0
        t_start = time.time()
        t_list = []
        keypoints_list = []
        while success:
            # img_loc = "frame%d.jpg" % count
            # cv2.imwrite(os.path.join(walking_data_path, "%d" % (index+1), "%d frame%d.jpg" % (index+1, count)), image)  # save frame as JPEG file
            datum = op.Datum()
            datum.cvInputData = image
            opWrapper.emplaceAndPop([datum])
            if len(datum.poseKeypoints.shape)==3:
                keypoints_list.append(datum.poseKeypoints[0, :, :])
                t_list.append(time.time() - t_start)
            video_writer.write(datum.cvOutputData)


            success, image = vidcap.read()
            # print('Read a new frame: ', success)
            count += 1

        json_filename = os.path.join('training_data',list_type[i],'keypoint_{0}.json'.format(date_time_now))
        data = {'body_keypoints': np.array(keypoints_list).tolist(),
             'time_stamps': t_list}

        with open(json_filename, 'w') as write_file:
             json.dump(data, write_file)














    # walking_data_path = r'training_data/walking/'
    # walking = glob.glob(os.path.join(walking_data_path, '*.mp4'))
    # # sys.path.append('../../python');
    #
    # for index, walking_video in enumerate(walking):
    #     dir = os.path.join(walking_data_path, "%d" % (index+1))
    #     if not os.path.exists(dir):
    #         os.mkdir(dir)
    #
    #     vidcap = cv2.VideoCapture(walking_video)
    #     height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    #     width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    #
    #     date_time_now = time.strftime('%Y-%m-%d-%H%M%S', time.localtime())
    #     video_out_file = 'training_data/walking/video_{0}.avi'.format(date_time_now)
    #     video_writer = cv2.VideoWriter(video_out_file, cv2.VideoWriter_fourcc(*'XVID'), 15, (int(width), int(height)))
    #
    #     success, image = vidcap.read()
    #     count = 0
    #     t_start = time.time()
    #     t_list = []
    #     keypoints_list = []
    #     while success:
    #         # img_loc = "frame%d.jpg" % count
    #         # cv2.imwrite(os.path.join(walking_data_path, "%d" % (index+1), "%d frame%d.jpg" % (index+1, count)), image)  # save frame as JPEG file
    #         datum = op.Datum()
    #         datum.cvInputData = image
    #         opWrapper.emplaceAndPop([datum])
    #         keypoints_list.append(datum.poseKeypoints[0, :, :])
    #         t_list.append(time.time() - t_start)
    #         video_writer.write(datum.cvOutputData)
    #
    #
    #         success, image = vidcap.read()
    #         # print('Read a new frame: ', success)
    #         count += 1
    #
    #     json_filename = 'training_data/walking/keypoint_{0}.json'.format(date_time_now)
    #     data = {'body_keypoints': np.array(keypoints_list).tolist(),
    #          'time_stamps': t_list}
    #
    #     with open(json_filename, 'w') as write_file:
    #          json.dump(data, write_file)


    # for index in range(len(walking)):
    #     date_time_now = time.strftime('%Y-%m-%d-%H%M%S', time.localtime())
    #     video_out_file = 'training_data/walking/video_{0}.avi'.format(date_time_now)
    #     video_writer = cv2.VideoWriter(video_out_file, cv2.VideoWriter_fourcc(*'XVID'), 15, (int(width), int(height)))
    #     dir = os.path.join(walking_data_path, "%d" % (index + 1))
    #     walking_image = glob.glob(os.path.join(dir, '*.jpg'))
    #
    #     t_start = time.time()
    #     t_list = []
    #     keypoints_list = []
    #
    #     for frame in walking_image:
    #         datum = op.Datum()
    #         datum.cvInputData = cv2.imread(frame)
    #         # opWrapper.emplaceAndPop([datum])
    #     # if not datum.poseKeypoints.shape:
    #     #     continue
    #         keypoints_list.append(datum.poseKeypoints[0, :, :])
    #         t_list.append(time.time() - t_start)
    #
    #     # cv2.imshow("Data Capture", datum.cvOutputData)
    #     # cv2.waitKey(1)
    #         video_writer.write(datum.cvOutputData)
    #     # key = cv2.waitKey(1)
    #     # if key == 27:
    #     #     break
    #     end = time.time()
    # # cap.release()
    #     video_writer.release()
    # # print("OpenPose demo successfully finished. Total time: " + str(end - t_start) + " seconds")
    #
    #     json_filename = 'training_data/walking/keypoint_{0}.json'.format(date_time_now)
    #     data = {'body_keypoints': np.array(keypoints_list).tolist(),
    #          'time_stamps': t_list}
    #
    #     with open(json_filename, 'w') as write_file:
    #          json.dump(data, write_file)