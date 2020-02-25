import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np
import json
import warnings
warnings.simplefilter('always', UserWarning)

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
    t_start = time.time()
    t_list = []
    keypoints_list = []






    cap = cv2.VideoCapture(0)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    date_time_now = time.strftime('%Y-%m-%d-%H%M%S', time.localtime())
    video_out_file = 'training_data/video_{0}.avi'.format(date_time_now)

    video_writer = cv2.VideoWriter(video_out_file, cv2.VideoWriter_fourcc(*'MJPG'), 15, (int(width), int(height)))

    # import glob
    # import cv2
    # imagefile = glob.glob('path to images from video')
    # # Process and display images
    # # for imagePath in imagePaths:
    # for image in imagefile:
    #     frame = cv2.imread(image)

    # this is used for real-time webcam
    while cap.isOpened():
        datum = op.Datum()
        # imageToProcess = cv2.imread(imagePath)
        ret, frame = cap.read()
        if ret:

            datum.cvInputData = frame
            opWrapper.emplaceAndPop([datum])

            # print("Body keypoints: \n" + str(datum.poseKeypoints))
            if not datum.poseKeypoints.shape:
                # warnings.warn("No keypoints detected!", category=None, stacklevel=1, source=None)
                continue

            keypoints_list.append(datum.poseKeypoints[0, :, :])
            t_list.append(time.time() - t_start)

            cv2.imshow("Data Capture", datum.cvOutputData)
            video_writer.write(datum.cvOutputData)
            key = cv2.waitKey(1)
            if key == 27:
                break


    end = time.time()
    cap.release()
    video_writer.release()
    print("OpenPose demo successfully finished. Total time: " + str(end - t_start) + " seconds")

    json_filename = 'training_data/keypoint_{0}.json'.format(date_time_now)
    data = {'body_keypoints': np.array(keypoints_list).tolist(),
            'time_stamps': t_list}

    with open(json_filename, 'w') as write_file:
        json.dump(data, write_file)