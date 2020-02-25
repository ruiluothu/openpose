import glob
import numpy as np
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import os
import json
from people_feature_generation import Person
import joblib
from sklearn.linear_model import LogisticRegression
# import statistics as stats

walking_data_path = r'training_data/walking/'
running_data_path = r'training_data/running/'
_default_save_model_name = 'action_svm.pkl'
_model_name_gpc = 'action_gpc.pkl'
_model_name_logit = 'action_logit.pkl'
_model_name_lda = 'action_lda.pkl'
_model_name_gnb = 'action_gnb.pkl'

class lda:
    def __init__(self, classes_=2, mu=None, sigma=None):
        self.classes_ = classes_
        self.mu = mu
        self.sigma = sigma

    def fit(self, X, y):
        y1 = y[y==np.unique(y)[0]]
        X1 = X[y==np.unique(y)[0],:]
        y2 = y[y==np.unique(y)[1]]
        X2 = X[y==np.unique(y)[1],:]
        self.classes_ = np.unique(y).shape[0]

        mu1, sigma1 = np.mean(X1, axis=0), np.std(X1, axis=0)
        mu2, sigma2 = np.mean(X2, axis=0), np.std(X2, axis=0)
        self.mu = np.concatenate((mu1.reshape(-1,1), mu2.reshape(-1,1)), axis=1)
        self.sigma = np.concatenate((sigma1.reshape(-1,1), sigma2.reshape(-1,1)), axis=1)

        return self

    def predict_proba(self, X):
        prob = np.zeros((X.shape[0], self.classes_))
        for i in range(self.classes_):
            mu = self.mu[:,i]
            sigma = self.sigma[:,i]
            prob[:,i] = -np.sum((X-mu)**2/sigma**2, axis=1)

        return prob

    def predict(self, X):
        prob = np.zeros((X.shape[0], self.classes_))
        for i in range(self.classes_):
            mu = self.mu[:, i]
            sigma = self.sigma[:, i]
            prob[:, i] = -np.sum((X - mu) ** 2 / sigma ** 2, axis=1)

        return np.argmax(prob,axis=1)

    def score(self, X, y):

        return sum(self.predict(X)==y)/len(y)


def get_action_classifier(method='SVM', featuresize = 16, load_model_name=_default_save_model_name, *train):

    # import training data and begin training
    if load_model_name is None:
        walking_files = np.array(train)[:,0]
        running_files = np.array(train)[:,1]
        features_walking = np.zeros((0, featuresize))
        features_running = np.zeros((0, featuresize))

        for file in walking_files[0]:
            data, t = _read_training_data(file)
            features = Person.get_2d_angles(data, t)  # n*24
            features_walking = np.concatenate((features_walking, features), axis=0)

        for file in running_files[0]:
            data, t = _read_training_data(file)
            features = Person.get_2d_angles(data, t)  # n*24
            if len(features.shape) == 2:
                features_running = np.concatenate((features_running, features), axis=0)

        X_walking = features_walking
        X_running = features_running
        y_walking = np.zeros(X_walking.shape[0], dtype=int)
        y_running = np.ones(X_running.shape[0], dtype=int)

        X_training = np.vstack((X_walking, X_running))
        y_training = np.hstack((y_walking, y_running))

        if method == 'GPC':
            print('Using GPC')
            kernel = 1.0 * RBF(1.0)
            gpc = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(X_training, y_training)
            save_model_name = _model_name_gpc
            joblib.dump(gpc, save_model_name)
            return gpc
        elif method == 'SVM':
            print('Using SVM')
            clf = svm.SVC(gamma='scale', probability=True)
            clf.fit(X_training, y_training)
            save_model_name = _default_save_model_name
            joblib.dump(clf, save_model_name)
            return clf
        elif method == 'Logit':
            print('Using Logistic Regression')
            logit = LogisticRegression().fit(X_training, y_training)
            save_model_name = _model_name_logit
            joblib.dump(logit, save_model_name)
            return logit
        elif method == 'LDA':
            print('Using LDA')
            lda = lda().fit(X_training, y_training)
            save_model_name = _model_name_lda
            joblib.dump(lda, save_model_name)
            return lda
        # elif method == 'Gaussian NB':
        #     print('Using Gaussian Naive Bayes')

    else:
        if method == 'GPC':
            print('Loading Action GPC')
            model = joblib.load(_model_name_gpc)

        elif method == 'SVM':
            print('Loading Action SVM')
            model = joblib.load(_default_save_model_name)

        elif method == 'Logit':
            print('Loading Action Logit')
            model = joblib.load(_model_name_logit)

        elif method == 'LDA':
            print('Loading Action LDA')
            model = joblib.load(_model_name_lda)


    return model


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
