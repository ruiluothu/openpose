import glob
import numpy as np
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import os
import json
from people_feature_generation import Person
import joblib

# import warnings
# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore",category=FutureWarning)
#     import joblib, sklearn, sklearn.externals, sklearn.externals.joblib
# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn = warn
# need to change the size of feature based on situation

walking_data_path = r'training_data/walking/'
jogging_data_path = r'training_data/jogging/'
running_data_path = r'training_data/running/'

_default_save_model_name1 = 'action_svm_all.pkl'
_default_save_model_name2 = 'action_svm_2d.pkl'
_default_save_model_name3 = 'action_svm_3d.pkl'
# _default_save_model_name_no_running = 'action_svm_no_running.pkl'


def get_action_classifier_all(use_gpc=False, load_model_name=_default_save_model_name1):
    if load_model_name is None:
        # Import raw data from videos of walking_rh and not walking_rh
        jogging_files = glob.glob(os.path.join(jogging_data_path, '*.json'))
        walking_files = glob.glob(os.path.join(walking_data_path, '*.json'))
        running_files = glob.glob(os.path.join(running_data_path, '*.json'))

        features_walking = np.zeros((0, 48))
        features_jogging = np.zeros((0, 48))
        features_running = np.zeros((0, 48))

        for file in walking_files:
            data, t = _read_training_data(file)
            # feature1 = Person.get_2d_angles(data, t) # n*24
            feature2, feature1 = Person.get_angles(data, t) # n*24
            # feature1 = np.delete(feature1, false_list, axis = 0)
            feature = np.concatenate((feature1,feature2),axis=1) # n*48
            features_walking = np.concatenate((features_walking,feature),axis=0)

        # for file in jogging_files:
        #     data, t = _read_training_data(file)
        #     feature2, feature1 = Person.get_angles(data, t) # n*24
        #     feature = np.concatenate((feature1,feature2),axis=1) # n*48
        #     # data, t = _read_training_data(file)
        #     # feature1 = Person.get_2d_angles(data, t) # n*24
        #     # feature2, false_list = Person.get_3d_angles(data, t) # n*24
        #     # feature1 = np.delete(feature1, false_list, axis = 0)
        #     # feature = np.concatenate((feature1,feature2),axis=1) # n*48
        #     features_jogging = np.concatenate((features_jogging,feature),axis=0)

        for file in running_files:
            data, t = _read_training_data(file)
            feature2, feature1 = Person.get_angles(data, t) # n*24
            feature = np.concatenate((feature1,feature2),axis=1) # n*48
            # data, t = _read_training_data(file)
            # feature1 = Person.get_2d_angles(data, t) # n*24
            # feature2, false_list = Person.get_3d_angles(data, t) # n*24
            # feature1 = np.delete(feature1, false_list, axis = 0)
            # feature = np.concatenate((feature1,feature2),axis=1) # n*48
            features_running = np.concatenate((features_running,feature),axis=0)

        # data_jogging, t_jogging = _read_training_data(jogging_files)
        # data_walking, t_walking = _read_training_data(walking_files)
        # data_running, t_running = _read_training_data(running_files)
        # features_walking = Person.get_2d_angles(data_walking, t_walking)
        # features_jogging = Person.get_2d_angles(data_jogging, t_jogging)
        # features_running = Person.get_2d_angles(data_running, t_running)

        # Train an SVM on the data to distinguish between walking_rh and not
        X_walking = features_walking
        # X_jogging = features_jogging
        X_running = features_running
        # X_walking = features_walking[valid_rows_walking, :]
        # X_jogging = features_jogging[valid_rows_jogging, :]
        # X_running = features_running[valid_rows_running, :]
        y_walking = np.zeros(X_walking.shape[0], dtype=int)
        # y_jogging = np.ones(X_jogging.shape[0], dtype=int)
        y_running = np.ones(X_running.shape[0], dtype=int) * 1

        X = np.vstack((X_walking, X_running))
        y = np.hstack((y_walking, y_running))

        training_row_mask = np.ones(y.shape, dtype=bool)
        X_training = X[training_row_mask, :]
        y_training = y[training_row_mask]

        if use_gpc:
            print('Using GPC')
            kernel = 1.0 * RBF(1.0)
            gpc = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(X_training, y_training)
            return gpc
        else:
            print('Training Action SVM')
            clf = svm.SVC(gamma='scale', probability=True)
            clf.fit(X_training, y_training)
            save_model_name = _default_save_model_name1
            joblib.dump(clf, save_model_name)

    else:
        print('Loading Action SVM')
        # if load_model_name == _default_save_model_name:
        #     clf = joblib.load(_default_save_model_name_no_running)
        # else:
        clf = joblib.load(load_model_name)

    return clf


    # # origin code
    # # Import raw data from videos of walking_rh and not walking_rh
    # jogging_files = glob.glob(os.path.join(jogging_data_path, '*.json'))
    # walking_files = glob.glob(os.path.join(walking_data_path, '*.json'))
    # running_files = glob.glob(os.path.join(running_data_path, '*.json'))
    #
    # data_jogging, t_jogging = _read_training_data(jogging_files)
    # data_walking, t_walking = _read_training_data(walking_files)
    # data_running, t_running = _read_training_data(running_files)
    #
    # features_walking = Person.get_2d_angles(data_walking, t_walking)
    # features_jogging = Person.get_2d_angles(data_jogging, t_jogging)
    # features_running = Person.get_2d_angles(data_running, t_running)
    #
    # # valid_rows_walking = Person.get_valid_features_mask(features_walking)
    # # valid_rows_jogging = Person.get_valid_features_mask(features_jogging)
    # # valid_rows_running = Person.get_valid_features_mask(features_running)
    #
    # # Train an SVM on the data to distinguish between walking_rh and not
    # X_walking = features_walking
    # X_jogging = features_jogging
    # X_running = features_running
    # # X_walking = features_walking[valid_rows_walking, :]
    # # X_jogging = features_jogging[valid_rows_jogging, :]
    # # X_running = features_running[valid_rows_running, :]
    # y_jogging = np.zeros(X_jogging.shape[0], dtype=int)
    # y_walking = np.ones(X_walking.shape[0], dtype=int)
    # y_running = np.ones(X_running.shape[0], dtype=int) * 2
    #
    # X = np.vstack((X_walking, X_jogging, X_running))
    # y = np.hstack((y_walking, y_jogging, y_running))
    #
    # training_row_mask = np.ones(y.shape, dtype=bool)
    # X_training = X[training_row_mask, :]
    # y_training = y[training_row_mask]
    #
    # # X_test = X[~training_row_mask, :]
    # # y_test = y[~training_row_mask]
    #
    # if use_gpc:
    #     print('Using GPC')
    #     kernel = 1.0 * RBF(1.0)
    #     gpc = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(X_training, y_training)
    #     return gpc
    # else:
    #     if load_model_name is None:
    #         print('Training Action SVM')
    #         clf = svm.SVC(gamma='scale', probability=True)
    #         clf.fit(X_training, y_training)
    #         save_model_name = _default_save_model_name
    #         joblib.dump(clf, save_model_name)
    #     else:
    #         print('Loading Action SVM')
    #         # if load_model_name == _default_save_model_name:
    #         #     clf = joblib.load(_default_save_model_name_no_running)
    #         # else:
    #         clf = joblib.load(load_model_name)
    #
    #     return clf




def get_action_classifier_2d_only(use_gpc=False, load_model_name=_default_save_model_name2):
    if load_model_name is None:
        # Import raw data from videos of walking_rh and not walking_rh
        jogging_files = glob.glob(os.path.join(jogging_data_path, '*.json'))
        walking_files = glob.glob(os.path.join(walking_data_path, '*.json'))
        running_files = glob.glob(os.path.join(running_data_path, '*.json'))

        features_walking = np.zeros((0, 24))
        features_jogging = np.zeros((0, 48))
        features_running = np.zeros((0, 24))

        for file in walking_files:
            data, t = _read_training_data(file)
            # feature1 = Person.get_2d_angles(data, t) # n*24
            _, feature1 = Person.get_angles(data, t) # n*24
            # feature1 = np.delete(feature1, false_list, axis = 0)
            # feature = np.concatenate((feature1,feature2),axis=1) # n*48
            features_walking = np.concatenate((features_walking,feature1),axis=0)

        # for file in jogging_files:
        #     data, t = _read_training_data(file)
        #     feature2, feature1 = Person.get_angles(data, t) # n*24
        #     feature = np.concatenate((feature1,feature2),axis=1) # n*48
        #     # data, t = _read_training_data(file)
        #     # feature1 = Person.get_2d_angles(data, t) # n*24
        #     # feature2, false_list = Person.get_3d_angles(data, t) # n*24
        #     # feature1 = np.delete(feature1, false_list, axis = 0)
        #     # feature = np.concatenate((feature1,feature2),axis=1) # n*48
        #     features_jogging = np.concatenate((features_jogging,feature),axis=0)

        for file in running_files:
            data, t = _read_training_data(file)
            _, feature1 = Person.get_angles(data, t) # n*24
            # feature = np.concatenate((feature1,feature2),axis=1) # n*48
            # data, t = _read_training_data(file)
            # feature1 = Person.get_2d_angles(data, t) # n*24
            # feature2, false_list = Person.get_3d_angles(data, t) # n*24
            # feature1 = np.delete(feature1, false_list, axis = 0)
            # feature = np.concatenate((feature1,feature2),axis=1) # n*48
            features_running = np.concatenate((features_running,feature1),axis=0)

        # data_jogging, t_jogging = _read_training_data(jogging_files)
        # data_walking, t_walking = _read_training_data(walking_files)
        # data_running, t_running = _read_training_data(running_files)
        # features_walking = Person.get_2d_angles(data_walking, t_walking)
        # features_jogging = Person.get_2d_angles(data_jogging, t_jogging)
        # features_running = Person.get_2d_angles(data_running, t_running)

        # Train an SVM on the data to distinguish between walking_rh and not
        X_walking = features_walking
        # X_jogging = features_jogging
        X_running = features_running
        # X_walking = features_walking[valid_rows_walking, :]
        # X_jogging = features_jogging[valid_rows_jogging, :]
        # X_running = features_running[valid_rows_running, :]
        y_walking = np.zeros(X_walking.shape[0], dtype=int)
        # y_jogging = np.ones(X_jogging.shape[0], dtype=int)
        y_running = np.ones(X_running.shape[0], dtype=int) * 1

        X = np.vstack((X_walking, X_running))
        y = np.hstack((y_walking, y_running))

        training_row_mask = np.ones(y.shape, dtype=bool)
        X_training = X[training_row_mask, :]
        y_training = y[training_row_mask]

        if use_gpc:
            print('Using GPC')
            kernel = 1.0 * RBF(1.0)
            gpc = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(X_training, y_training)
            return gpc
        else:
            print('Training Action SVM')
            clf = svm.SVC(gamma='scale', probability=True)
            clf.fit(X_training, y_training)
            save_model_name = _default_save_model_name2
            joblib.dump(clf, save_model_name)

    else:
        print('Loading Action SVM')
        # if load_model_name == _default_save_model_name:
        #     clf = joblib.load(_default_save_model_name_no_running)
        # else:
        clf = joblib.load(load_model_name)

    return clf




def get_action_classifier_3d_only(use_gpc=False, load_model_name=_default_save_model_name3):
    if load_model_name is None:
        # Import raw data from videos of walking_rh and not walking_rh
        jogging_files = glob.glob(os.path.join(jogging_data_path, '*.json'))
        walking_files = glob.glob(os.path.join(walking_data_path, '*.json'))
        running_files = glob.glob(os.path.join(running_data_path, '*.json'))

        features_walking = np.zeros((0, 24))
        features_jogging = np.zeros((0, 48))
        features_running = np.zeros((0, 24))

        for file in walking_files:
            data, t = _read_training_data(file)
            # feature1 = Person.get_2d_angles(data, t) # n*24
            feature2, _ = Person.get_angles(data, t) # n*24
            # feature1 = np.delete(feature1, false_list, axis = 0)
            # feature = np.concatenate((feature1,feature2),axis=1) # n*48
            features_walking = np.concatenate((features_walking,feature2),axis=0)

        # for file in jogging_files:
        #     data, t = _read_training_data(file)
        #     feature2, feature1 = Person.get_angles(data, t) # n*24
        #     feature = np.concatenate((feature1,feature2),axis=1) # n*48
        #     # data, t = _read_training_data(file)
        #     # feature1 = Person.get_2d_angles(data, t) # n*24
        #     # feature2, false_list = Person.get_3d_angles(data, t) # n*24
        #     # feature1 = np.delete(feature1, false_list, axis = 0)
        #     # feature = np.concatenate((feature1,feature2),axis=1) # n*48
        #     features_jogging = np.concatenate((features_jogging,feature),axis=0)

        for file in running_files:
            data, t = _read_training_data(file)
            feature2, _ = Person.get_angles(data, t) # n*24
            # feature = np.concatenate((feature1,feature2),axis=1) # n*48
            # data, t = _read_training_data(file)
            # feature1 = Person.get_2d_angles(data, t) # n*24
            # feature2, false_list = Person.get_3d_angles(data, t) # n*24
            # feature1 = np.delete(feature1, false_list, axis = 0)
            # feature = np.concatenate((feature1,feature2),axis=1) # n*48
            features_running = np.concatenate((features_running,feature2),axis=0)

        # data_jogging, t_jogging = _read_training_data(jogging_files)
        # data_walking, t_walking = _read_training_data(walking_files)
        # data_running, t_running = _read_training_data(running_files)
        # features_walking = Person.get_2d_angles(data_walking, t_walking)
        # features_jogging = Person.get_2d_angles(data_jogging, t_jogging)
        # features_running = Person.get_2d_angles(data_running, t_running)

        # Train an SVM on the data to distinguish between walking_rh and not
        X_walking = features_walking
        # X_jogging = features_jogging
        X_running = features_running
        # X_walking = features_walking[valid_rows_walking, :]
        # X_jogging = features_jogging[valid_rows_jogging, :]
        # X_running = features_running[valid_rows_running, :]
        y_walking = np.zeros(X_walking.shape[0], dtype=int)
        # y_jogging = np.ones(X_jogging.shape[0], dtype=int)
        y_running = np.ones(X_running.shape[0], dtype=int) * 1

        X = np.vstack((X_walking, X_running))
        y = np.hstack((y_walking, y_running))

        training_row_mask = np.ones(y.shape, dtype=bool)
        X_training = X[training_row_mask, :]
        y_training = y[training_row_mask]

        if use_gpc:
            print('Using GPC')
            kernel = 1.0 * RBF(1.0)
            gpc = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(X_training, y_training)
            return gpc
        else:
            print('Training Action SVM')
            clf = svm.SVC(gamma='scale', probability=True)
            clf.fit(X_training, y_training)
            save_model_name = _default_save_model_name3
            joblib.dump(clf, save_model_name)

    else:
        print('Loading Action SVM')
        # if load_model_name == _default_save_model_name:
        #     clf = joblib.load(_default_save_model_name_no_running)
        # else:
        clf = joblib.load(load_model_name)

    return clf



def _read_training_data(file_names):
    training_data = np.zeros([0, 25, 3])
    t_log = np.zeros([0])
    if isinstance(file_names, str):
        with open(file_names) as json_file:
            data = json.load(json_file)
            pos_data_arr = np.array(data['body_keypoints'])
            t_data_arr = np.array(data['time_stamps'])
            training_data = np.concatenate((training_data, pos_data_arr), axis=0)
            t_log = np.concatenate((t_log, t_data_arr), axis=0)
    else:
        for file_name in file_names:
            with open(file_name) as json_file:
                data = json.load(json_file)
                pos_data_arr = np.array(data['body_keypoints'])
                t_data_arr = np.array(data['time_stamps'])
                training_data = np.concatenate((training_data, pos_data_arr), axis=0)
                t_log = np.concatenate((t_log, t_data_arr), axis=0)

    return training_data, t_log