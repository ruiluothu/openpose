import numpy as np
import math
from enum import Enum


class Joints(Enum):
    nose = 0
    neck = 1
    r_shoulder = 2
    r_elbow = 3
    r_wrist = 4
    l_shoulder = 5
    l_elbow = 6
    l_wrist = 7
    mid_hip = 8
    r_hip = 9
    r_knee = 10
    r_ankle = 11
    l_hip = 12
    l_knee = 13
    l_ankle = 14
    r_eye = 15
    l_eye = 16
    r_ear = 17
    l_ear = 18
    l_big_toe = 19
    l_small_toe = 20
    l_heel = 21
    r_big_toe = 22
    r_small_toe = 23
    r_heel = 24
    r_background = 25


angle_index_pair = np.array([[Joints.neck.value,Joints.l_shoulder.value,Joints.l_shoulder.value,Joints.l_elbow.value], \
                             [Joints.l_shoulder.value,Joints.l_elbow.value,Joints.l_elbow.value,Joints.l_wrist.value], \
                             [Joints.neck.value,Joints.r_shoulder.value,Joints.r_shoulder.value,Joints.r_elbow.value],\
                             [Joints.r_shoulder.value,Joints.r_elbow.value,Joints.r_elbow.value,Joints.r_wrist.value], \
                             [Joints.mid_hip.value,Joints.l_hip.value,Joints.l_hip.value,Joints.l_knee.value], \
                             [Joints.l_hip.value,Joints.l_knee.value,Joints.l_knee.value,Joints.l_ankle.value], \
                             [Joints.mid_hip.value,Joints.r_hip.value,Joints.r_hip.value,Joints.r_knee.value], \
                             [Joints.r_hip.value,Joints.r_knee.value,Joints.r_knee.value,Joints.r_ankle.value]])
keypoints_loc = np.unique(angle_index_pair.flatten())

class ActionLabel(Enum):
    walking = 0
    jogging = 1
    running = 2


class Person:
    _person_ID = 0

    def __init__(self, action_confidence_thresh=0.8, _ratio = [0.4428,    0.6479,    0.5616,    0.4104,    0.8207,    0.7991]):
        # self.pose_keypoints = np.array([25, 3])
        self.keypoints_buffer = np.zeros([0, 25, 3])
        self.t_stamps = np.zeros(3)
        self.action_buffer = None
        self.action_confidence_threshold = action_confidence_thresh
        self.mid_hip_position = np.array([0, 0])
        self.ID = Person._person_ID
        self.torso_length = 0.463
        self.ratio = np.asarray(_ratio)
        self.keypoints_loc = np.unique(angle_index_pair.flatten())
        self.focal_length = 0.1
        Person._person_ID += 1

    # @staticmethod
    # def get_2d_angles(keypoints_data, t_log, index_pair=angle_index_pair):
    #     keypoints_loc = np.unique(index_pair.flatten())
    #     jump=[i for i in range(1, np.diff(t_log).shape[0] - 1) if
    #      np.diff(t_log)[i] > 3 * np.diff(t_log)[i - 1] and np.diff(t_log)[i] > 3 * np.diff(t_log)[i + 1]]
    #     total_angle_2d = np.zeros((0,index_pair.shape[0]))
    #     total_angle_2d_dot = np.zeros((0,index_pair.shape[0]))
    #     total_angle_2d_ddot = np.zeros((0,index_pair.shape[0]))
    #     # total_angle_2d = np.zeros((keypoints_data.shape[0]-len(jump),0))
    #     # total_angle_2d_dot = np.zeros((keypoints_data.shape[0] - len(jump), 0))
    #     # total_angle_2d_ddot = np.zeros((keypoints_data.shape[0] - len(jump), 0))
    #     jump.insert(0,-1)
    #     jump.append(t_log.shape[0])
    #
    #     for index in range(len(jump)-1):
    #         data = keypoints_data[jump[index]+1:jump[index+1],:,:]
    #         t = t_log[jump[index]+1:jump[index+1]]
    #         valid_mask = np.all(~np.any(data[:, keypoints_loc, :] == 0, axis=1), axis=1)
    #         data = data[valid_mask,:,:]
    #         if data.shape[0]>3:
    #             t = t[valid_mask]
    #             angle_2d = np.zeros((len(t),0))
    #             for i in range(index_pair.shape[0]):
    #                 vector1 = data[:, index_pair[i][1], 0:2] - data[:, index_pair[i][0], 0:2]
    #                 vector2 = data[:, index_pair[i][3], 0:2] - data[:, index_pair[i][2], 0:2]
    #                 angle = np.arccos(np.sum(vector1 * vector2, axis=1) / np.sqrt(
    #                 np.sum(vector1 * vector1, axis=1) * np.sum(vector2 * vector2, axis=1)))
    #             # angle = angle_all[~np.isnan(angle_all)]
    #                 angle_2d = np.concatenate((angle_2d, angle.reshape(-1, 1)), axis=1)
    #             angle_2d_dot = np.gradient(angle_2d, t, axis=0)
    #             angle_2d_ddot = np.gradient(angle_2d_dot, t, axis=0)
    #             total_angle_2d = np.concatenate((total_angle_2d,angle_2d),axis=0)
    #             total_angle_2d_dot = np.concatenate((total_angle_2d_dot,angle_2d_dot),axis=0)
    #             total_angle_2d_ddot = np.concatenate((total_angle_2d_ddot, angle_2d_ddot), axis=0)
    #
    #     return np.concatenate((total_angle_2d, total_angle_2d_dot, total_angle_2d_ddot), axis=1)







        # t_log = np.delete(t_log,jump)
        # keypoints_data = np.delete(keypoints_data,jump,axis=0)
        #
        # if keypoints_data.shape[0] != 3:
        #     valid_mask = np.all(~np.any(keypoints_data[:,keypoints_loc,:]==0,axis=1),axis=1)
        #     keypoints_data = keypoints_data[valid_mask,:,:]
        #     t_log = t_log[valid_mask]
        #     angle_2d = np.zeros((len(t_log),0))
        #     for i in range(index_pair.shape[0]):
        #         vector1 = keypoints_data[:,index_pair[i][1],0:2]-keypoints_data[:,index_pair[i][0],0:2]
        #         vector2 = keypoints_data[:,index_pair[i][3],0:2]-keypoints_data[:,index_pair[i][2],0:2]
        #         angle = np.arccos(np.sum(vector1*vector2,axis=1)/np.sqrt(np.sum(vector1*vector1,axis=1)*np.sum(vector2*vector2,axis=1)))
        #     # angle = angle_all[~np.isnan(angle_all)]
        #         angle_2d = np.concatenate((angle_2d, angle.reshape(-1,1)), axis=1)
        #     angle_2d_dot = np.gradient(angle_2d, t_log, axis=0)
        #     angle_2d_ddot = np.gradient(angle_2d_dot, t_log, axis=0)
        #
        # else:
        #     # if ~np.any(keypoints_data[:, keypoints_loc, :] == 0):
        #     t_log = t_log.flatten()
        #     angle_2d = np.zeros((len(t_log), 0))
        #     for i in range(index_pair.shape[0]):
        #         vector1 = keypoints_data[:, index_pair[i][1], 0:2] - keypoints_data[:, index_pair[i][0], 0:2]
        #         vector2 = keypoints_data[:, index_pair[i][3], 0:2] - keypoints_data[:, index_pair[i][2], 0:2]
        #         angle = np.arccos(np.sum(vector1 * vector2, axis=1) / np.sqrt(np.sum(vector1 * vector1, axis=1) * np.sum(vector2 * vector2, axis=1)))
        #         angle_2d = np.concatenate((angle_2d, angle.reshape(-1, 1)), axis=1)
        #     angle_2d_dot = np.gradient(angle_2d, t_log, axis=0)
        #     angle_2d_ddot = np.gradient(angle_2d_dot, t_log, axis=0)


            # valid_mask = np.all(~np.any(keypoints_data[:, keypoints_loc, :] == 0, axis=1), axis=0)
            # keypoints_data = keypoints_data[valid_mask, :, :]
            # t_log = t_log[valid_mask]
            # t_log = t_log.flatten()
            # angle_2d = np.zeros((len(t_log), 0))
            # for i in range(index_pair.shape[0]):
            #     vector1 = keypoints_data[:,index_pair[i][1],0:2]-keypoints_data[:,index_pair[i][0],0:2]
            #     vector2 = keypoints_data[:,index_pair[i][3],0:2]-keypoints_data[:,index_pair[i][2],0:2]
            #     angle = np.arccos(np.sum(vector1*vector2,axis=1)/np.sqrt(np.sum(vector1*vector1,axis=1)*np.sum(vector2*vector2,axis=1)))
            # # angle = angle_all[~np.isnan(angle_all)]
            #     angle_2d = np.concatenate((angle_2d, angle.reshape(-1,1)), axis=1)
            # angle_2d_dot = np.gradient(angle_2d, t_log, axis=0)
            # angle_2d_ddot = np.gradient(angle_2d_dot, t_log, axis=0)
        # return np.concatenate((angle_2d_dot, angle_2d_ddot), axis=1)



    # def update_pose_keypoints(self, keypoints, t_stamp):
    #     self.keypoints_buffer[...] = np.roll(self.keypoints_buffer, 1, axis=0)
    #     self.keypoints_buffer[0, :, :] = keypoints
    #
    #     self.t_stamps[1:len(self.t_stamps)] = self.t_stamps[0:(len(self.t_stamps)-1)]
    #     self.t_stamps[0] = t_stamp
    #     return self

    def action_classification_all(keypoints, time, classifier, use_probability=True):
        # features1 = Person.get_2d_angles(keypoints, time, index_pair=angle_index_pair)
        # features2 = Person.get_3d_angles(keypoints, time, index_pair=angle_index_pair)
        feature2, feature1 = Person.get_angles(keypoints, time, index_pair=angle_index_pair)
        features = np.concatenate((feature1,feature2),axis=1)
        label = np.zeros(0)
        for i in range(features.shape[0]):
            label_probabilities = np.zeros(len(classifier.classes_))
            if (not np.any(np.isnan(features[i,:]))):
                if use_probability:
                    label_probabilities = classifier.predict_proba(features[i,:].reshape(1,-1))[0]
                    label_value = np.argmax(label_probabilities)
                else:
                    label_value = classifier.predict(features[i,:].reshape(1,features.shape[1]))[0]
                label = np.concatenate((label, np.array(ActionLabel(label_value).value).flatten()),axis=0)

        return label




    def action_classification_2d(keypoints, time, classifier, use_probability=True):
        # features1 = Person.get_2d_angles(keypoints, time, index_pair=angle_index_pair)
        # features2 = Person.get_3d_angles(keypoints, time, index_pair=angle_index_pair)
        _, feature1 = Person.get_angles(keypoints, time, index_pair=angle_index_pair)
        features = feature1
        label = np.zeros(0)
        for i in range(features.shape[0]):
            label_probabilities = np.zeros(len(classifier.classes_))
            if (not np.any(np.isnan(features[i,:]))):
                if use_probability:
                    label_probabilities = classifier.predict_proba(features[i,:].reshape(1,-1))[0]
                    label_value = np.argmax(label_probabilities)
                else:
                    label_value = classifier.predict(features[i,:].reshape(1,features.shape[1]))[0]
                label = np.concatenate((label, np.array(ActionLabel(label_value).value).flatten()),axis=0)

        return label




    def action_classification_3d(keypoints, time, classifier, use_probability=True):
        # features1 = Person.get_2d_angles(keypoints, time, index_pair=angle_index_pair)
        # features2 = Person.get_3d_angles(keypoints, time, index_pair=angle_index_pair)
        feature2, _ = Person.get_angles(keypoints, time, index_pair=angle_index_pair)
        features = feature2
        label = np.zeros(0)
        for i in range(features.shape[0]):
            label_probabilities = np.zeros(len(classifier.classes_))
            if (not np.any(np.isnan(features[i,:]))):
                if use_probability:
                    label_probabilities = classifier.predict_proba(features[i,:].reshape(1,-1))[0]
                    label_value = np.argmax(label_probabilities)
                else:
                    label_value = classifier.predict(features[i,:].reshape(1,features.shape[1]))[0]
                label = np.concatenate((label, np.array(ActionLabel(label_value).value).flatten()),axis=0)

        return label





    def update_action_classification(keypoints, time, classifier, use_probability=True):
        label = ActionLabel.walking
        label_probabilities = np.zeros(len(classifier.classes_))
        # if np.sum(self.t_stamps == 0) <= 1:
        features = Person.get_2d_angles(keypoints, time, index_pair=angle_index_pair)

            # Check here for nans in the features - must assume action is Other if there are any
        if (not np.any(np.isnan(features))):
            if use_probability:
                label_probabilities = classifier.predict_proba(features)[0]
                label_value = np.argmax(label_probabilities)
            else:
                label_value = classifier.predict(features)[0]
            label = ActionLabel(label_value)

        # if use_probability:
        #     self._update_action_buffers(label, use_probabilities=True, probabilities=label_probabilities)
        # else:
        #     self._update_action_buffers(label)

        return label


    @staticmethod
    def get_angles(keypoints_data, t_log, index_pair=angle_index_pair):  # n*25*3
        keypoints_loc = np.unique(index_pair.flatten())
    # 先不加jump point检测
        # jump=[i for i in range(1, np.diff(t_log).shape[0] - 1) if
    #  np.diff(t_log)[i] > 3 * np.diff(t_log)[i - 1] and np.diff(t_log)[i] > 3 * np.diff(t_log)[i + 1]]
    # t_log = np.delete(t_log,jump)
    # keypoints_data = np.delete(keypoints_data,jump,axis=0)
        jump=[i for i in range(1, np.diff(t_log).shape[0] - 1) if
         np.diff(t_log)[i] > 3 * np.diff(t_log)[i - 1] and np.diff(t_log)[i] > 3 * np.diff(t_log)[i + 1]]
        jump.insert(0,-1)
        jump.append(t_log.shape[0])
        total_angle_2d = np.zeros((0,index_pair.shape[0]))
        total_angle_2d_dot = np.zeros((0,index_pair.shape[0]))
        total_angle_2d_ddot = np.zeros((0,index_pair.shape[0]))
        total_angle_3d = np.zeros((0,index_pair.shape[0]))
        total_angle_3d_dot = np.zeros((0,index_pair.shape[0]))
        total_angle_3d_ddot = np.zeros((0,index_pair.shape[0]))
        for index in range(len(jump)-1):
            data = keypoints_data[jump[index]+1:jump[index+1],:,:]
            t = t_log[jump[index]+1:jump[index+1]]
            valid_mask = np.all(~np.any(data[:, keypoints_loc, :] == 0, axis=1), axis=1)
            data = data[valid_mask, :, :] # n*25*3
            data = data[:, keypoints_loc, :] # n*14*3
            t = t[valid_mask]
            keypoints_data, false_list = from_2d_to_3d(data)  # n*14*3
            t = np.delete(t, false_list, 0)
            angle_2d = np.zeros((len(t), 0))
            angle_3d = np.zeros((len(t), 0))
            index_pair = index_pair - 1
            if len(t) !=0:
                for i in range(index_pair.shape[0]):
                    vector1 = keypoints_data[:, index_pair[i][1], 0:3] - keypoints_data[:, index_pair[i][0], 0:3]
                    vector2 = keypoints_data[:, index_pair[i][3], 0:3] - keypoints_data[:, index_pair[i][2], 0:3]
                    angle = np.arccos(np.sum(vector1 * vector2, axis=1) / np.sqrt(
                    np.sum(vector1 * vector1, axis=1) * np.sum(vector2 * vector2, axis=1)))
                # angle = angle_all[~np.isnan(angle_all)]
                    angle_3d = np.concatenate((angle_3d, angle.reshape(-1, 1)), axis=1)
                    vector1 = keypoints_data[:, index_pair[i][1], 0:2] - keypoints_data[:, index_pair[i][0], 0:2]
                    vector2 = keypoints_data[:, index_pair[i][3], 0:2] - keypoints_data[:, index_pair[i][2], 0:2]
                    angle = np.arccos(np.sum(vector1 * vector2, axis=1) / np.sqrt(
                    np.sum(vector1 * vector1, axis=1) * np.sum(vector2 * vector2, axis=1)))
                # angle = angle_all[~np.isnan(angle_all)]
                    angle_2d = np.concatenate((angle_2d, angle.reshape(-1, 1)), axis=1)

                angle_2d_dot = np.gradient(angle_2d, t, axis=0)
                angle_2d_ddot = np.gradient(angle_2d_dot, t, axis=0)
                angle_3d_dot = np.gradient(angle_3d, t, axis=0)
                angle_3d_ddot = np.gradient(angle_3d_dot, t, axis=0)
                total_angle_3d = np.concatenate((total_angle_3d, angle_3d), axis=0)
                total_angle_3d_dot = np.concatenate((total_angle_3d_dot, angle_3d_dot), axis=0)
                total_angle_3d_ddot = np.concatenate((total_angle_3d_ddot, angle_3d_ddot), axis=0)
                total_angle_2d = np.concatenate((total_angle_2d,angle_2d),axis=0)
                total_angle_2d_dot = np.concatenate((total_angle_2d_dot,angle_2d_dot),axis=0)
                total_angle_2d_ddot = np.concatenate((total_angle_2d_ddot, angle_2d_ddot), axis=0)


        # valid_mask = np.all(~np.any(keypoints_data[:, keypoints_loc, :] == 0, axis=1), axis=1)
        # keypoints_data = keypoints_data[valid_mask, :, :]  # n*14*3
        # t_log = t_log[valid_mask]
        # keypoints_data, false_list = from_2d_to_3d(keypoints_data)  # n*14*3
        # t_log = np.delete(t_log,false_list,0)
        # angle_3d = np.zeros((len(t_log), 0))
        # for i in range(index_pair.shape[0]):
        #     vector1 = keypoints_data[:, index_pair[i][1], 0:3] - keypoints_data[:, index_pair[i][0], 0:3]
        #     vector2 = keypoints_data[:, index_pair[i][3], 0:3] - keypoints_data[:, index_pair[i][2], 0:3]
        #     angle = np.arccos(np.sum(vector1 * vector2, axis=1) / np.sqrt(
        #     np.sum(vector1 * vector1, axis=1) * np.sum(vector2 * vector2, axis=1)))
        # # angle = angle_all[~np.isnan(angle_all)]
        #     angle_3d = np.concatenate((angle_3d, angle.reshape(-1, 1)), axis=1)
        # angle_3d_dot = np.gradient(angle_3d, t_log, axis=0)
        # angle_3d_ddot = np.gradient(angle_3d_dot, t_log, axis=0)
        # return np.concatenate((total_angle_3d, total_angle_3d_dot, total_angle_3d_ddot), axis=1), false_list
        # return np.concatenate((angle_3d, angle_3d_dot, angle_3d_ddot), axis=1), false_list
        return np.concatenate((total_angle_3d, total_angle_3d_dot, total_angle_3d_ddot), axis=1), np.concatenate((total_angle_2d, total_angle_2d_dot, total_angle_2d_ddot), axis=1)


# def from_2d_to_3d(keypoints_data,Person().ratio,Person().torso_length,Person().keypoints_loc):
def from_2d_to_3d(keypoints_data,focal_length=Person().focal_length,ratio=Person().ratio,torso_length=Person().torso_length):
    focal_lambda = focal_length
    list_3d = np.zeros((0, 14, 3))
    L1 = torso_length*ratio
    length_body = np.vstack((L1[[0,1,2]],L1[[0,1,2]],L1[[3,4,5]],L1[[3,4,5]]))
    false_list = []
    for count, coord_2d in enumerate(keypoints_data[:, :, :][:,:, 0: 2]):  # coord_2d: 25*2
        coord_2d = coord_2d/500
        lt = np.linalg.norm(coord_2d[Joints.mid_hip.value] - coord_2d[Joints.neck.value])
        k0 = torso_length / lt
        index_leftarm = np.array([Joints.neck.value, Joints.l_shoulder.value, Joints.l_elbow.value, Joints.l_wrist.value])-1
        index_rightarm = np.array([Joints.neck.value, Joints.r_shoulder.value, Joints.r_elbow.value, Joints.r_wrist.value])-1
        index_leftleg = np.array([Joints.mid_hip.value, Joints.l_hip.value, Joints.l_knee.value, Joints.l_ankle.value])-1
        index_rightleg = np.array([Joints.mid_hip.value, Joints.r_hip.value, Joints.r_knee.value, Joints.r_ankle.value])-1
        index_body = np.vstack((index_leftarm, index_rightarm, index_leftleg, index_rightleg))
        # k_list = np.zeros((4,0))
        k_list = []
        for index in range(0,4):
            index_part = index_body[index, :]
            k_old = np.array([k0])
            L = length_body[index]
            for i in range(0,index_part.shape[0]-1):
                startpoint = coord_2d[index_part[i], :]
                endpoint = coord_2d[index_part[i + 1], :]
                x1 = startpoint[0]
                y1 = startpoint[1]
                x2 = endpoint[0]
                y2 = endpoint[1]
                if len(k_old.shape) > 1:
                    k_temp = np.zeros((0, k_old.shape[1] + 1))
                    for j in range(k_old.shape[0]):
                        k_new = np.roots([x2 ** 2 + y2 ** 2 +
                                  focal_lambda ** 2, -2 * k_old[j, 0] * (x1 * x2 + y1 * y2 + focal_lambda ** 2), k_old[j,0] ** 2 * (x1 ** 2 + y1 ** 2 + focal_lambda ** 2) - L[3-i] ** 2])
                        k_new = k_new[np.imag(k_new)==0]
                        k_temp = np.concatenate(
                    (k_temp, np.hstack((k_new.reshape(-1, 1), np.repeat(k_old[j, :].reshape(1,-1), k_new.shape[0], axis=0)))),
                    axis=0)
                    k_old = k_temp
                else:
                    k_new = np.roots([x2 ** 2 + y2 ** 2 +
                              focal_lambda ** 2, -2 * k_old * (x1 * x2 + y1 * y2 + focal_lambda ** 2), k_old ** 2 * (
                                          x1 ** 2 + y1 ** 2 + focal_lambda ** 2) - L[2-i] ** 2])
                    k_new = k_new[np.imag(k_new) == 0]
                    k_old = np.hstack((k_new.reshape(-1, 1), np.repeat(k_old, k_new.shape[0], axis=0).reshape(-1,1)))  # k_old: (8 or 4)*4
            k_list.append(k_old)  # k_list: 4* [(8 or 4)*4]

# check the 3d coords corresponding to these k values, in the order of 4 body parts
        coord_3d = []
        for i in range(len(k_list)):
            index_part = index_body[i, :]
            coord = coord_2d[index_part, :]  # 4*2
            coord_expand = np.concatenate((coord, focal_lambda *np.ones((4, 1))),axis=1)  # 4*3
            coord_part_3d = np.zeros((0, 4, 3))
            for j in range(k_list[i].shape[0]):
                coord_expand_3d = k_list[i][j].reshape(-1, 1) * coord_expand  # 4*3
                coord_part_3d = np.concatenate((coord_part_3d, coord_expand_3d.reshape(1, 4, 3)), axis=0)  # (8 or 4)*4*3
            coord_3d.append(coord_part_3d)
            # if i == 0:
            #     coord_3d = np.array([coord_part_3d])
            # else:
            #     coord_3d = np.array([coord_3d, coord_part_3d])  # this coord_3d is actually the set of all possible values,
        # it has 4 dimension, 4*(8 or 4*4*3), the first 4 represents four part: (left,right) x (arm,leg)
        if any([v.shape[0]==0 for v in coord_3d]):
            false_list.append(count)
        else:
            coord_3d = verify(coord_3d)  # k*14*3
            coord_3d = balance(coord_3d)  # 14*3
            list_3d = np.concatenate((list_3d, coord_3d.reshape(1, 14, 3)), axis=0)
    return list_3d, false_list


def verify(coord_3d):
        O1 = np.real(coord_3d[0][0, -1, :])
        O2 = np.real(coord_3d[-1][0, -1, :])
        good_list = np.zeros((0, 14, 3))
        wait_list = np.zeros((0, 14, 3))
        wait_list_point = np.zeros(0)
        for coord_leftarm in np.real(coord_3d[0][:, 0:-1, :]):  # 8 or 4 *3*3
            for coord_rightarm in np.real(coord_3d[1][:, 0:-1, :]):
                for coord_leftleg in np.real(coord_3d[2][:, 0:-1, :]):
                    for coord_rightleg in np.real(coord_3d[3][:, 0:-1, :]):
                        C1 = coord_leftarm[0, :]
                        B1 = coord_leftarm[1, :]
                        A1 = coord_leftarm[2, :]
                        F1 = coord_rightarm[0, :]
                        E1 = coord_rightarm[1, :]
                        D1 = coord_rightarm[2, :]
                        C2 = coord_leftleg[0, :]
                        B2 = coord_leftleg[1, :]
                        A2 = coord_leftleg[2, :]
                        F2 = coord_rightleg[0, :]
                        E2 = coord_rightleg[1, :]
                        D2 = coord_rightleg[2, :]
                        v1, w1 = uppercheck(O1, O2, A1, B1, C1, D1, E1, F1, A2, B2, C2, D2, E2, F2)
                        v2, w2 = lowercheck(O1, O2, A1, B1, C1, D1, E1, F1, A2, B2, C2, D2, E2, F2)
                        v3, w3 = crosscheck(O1, O2, A1, B1, C1, D1, E1, F1, A2, B2, C2, D2, E2, F2)
                        if v1 and v2 and v3:
                            good_3d = np.vstack((O1, O2, A1, B1, C1, D1, E1, F1, A2, B2, C2, D2, E2, F2)).T  # 14*3
                            good_list = np.concatenate((good_list, good_3d.reshape(1, 14, 3)),
                                                       axis=0)
                        else:
                            wait = np.vstack((O1, O2, A1, B1, C1, D1, E1, F1, A2, B2, C2, D2, E2, F2)).T  # 14*3
                            wait_point = np.array([w1+w2+w3])
                            wait_list = np.concatenate((wait_list,wait.reshape(1,14,3)),axis=0)
                            wait_list_point = np.concatenate((wait_list_point,wait_point),axis=0)
        if good_list.shape[0] == 0:
            good_list = wait_list[np.argmax(wait_list_point),:,:].reshape(1,14,3)
                        # if uppercheck(O1, O2, A1, B1, C1, D1, E1, F1, A2, B2, C2, D2, E2, F2) + lowercheck(O1, O2, A1, B1,
                        #                                                                                  C1, D1, E1, F1,
                        #                                                                                  A2, B2, C2, D2,
                        #                                                                                  E2,
                        #                                                                                  F2) + crosscheck(
                        #     O1, O2, A1, B1, C1, D1, E1, F1, A2, B2, C2, D2, E2, F2) >= 18:
                        # if uppercheck(O1, O2, A1, B1, C1, D1, E1, F1, A2, B2, C2, D2, E2, F2) and lowercheck(O1, O2, A1, B1,
                        #                                                                                  C1, D1, E1, F1,
                        #                                                                                  A2, B2, C2, D2,
                        #                                                                                  E2,
                        #                                                                                  F2) and crosscheck(
                        #     O1, O2, A1, B1, C1, D1, E1, F1, A2, B2, C2, D2, E2, F2):
        return good_list  # k*14*3


def uppercheck(O1, O2, A1, B1, C1, D1, E1, F1, A2, B2, C2, D2, E2, F2):
        O1, O2, A1, B1, C1, D1, E1, F1, A2, B2, C2, D2, E2, F2 = np.real(O1),np.real(O2),np.real(A1),np.real(B1),np.real(C1),np.real(D1),np.real(E1),np.real(F1),np.real(A2),np.real(B2),np.real(C2),np.real(D2),np.real(E2),np.real(F2)
        norm_vec = -np.cross(D1 - O2, A1 - O2)
        norm_vec = norm_vec / np.linalg.norm(norm_vec)
        norm_vec1 = np.cross(B1 - O1, A1 - O1)
        norm_vec1 = norm_vec1 / np.linalg.norm(norm_vec1)
        norm_vec2 = np.cross(D1 - O1, E1 - O1)
        norm_vec2 = norm_vec2 / np.linalg.norm(norm_vec2)

        angle1 = np.arccos(np.dot(D1 - O1, O1 - A1) / np.linalg.norm(D1 - O1) / np.linalg.norm(O1 - A1))
        angle2 = np.arccos(np.dot(A1 - O1, norm_vec) / np.linalg.norm(A1 - O1))
        angle3 = np.arccos(np.dot(D1 - O1, norm_vec) / np.linalg.norm(D1 - O1))
        angle4 = np.arccos(np.dot(B1 - A1, norm_vec) / np.linalg.norm(B1 - A1))
        angle5 = np.arccos(np.dot(E1 - D1, norm_vec) / np.linalg.norm(E1 - D1))
        angle6 = np.arccos(np.dot(A1 - O1, B1 - A1) / np.linalg.norm(A1 - O1) / np.linalg.norm(B1 - A1))
        angle7 = np.arccos(np.dot(D1 - O1, E1 - D1) / np.linalg.norm(D1 - O1) / np.linalg.norm(E1 - D1))
        angle8 = np.arccos(np.dot(C1 - B1, B1 - A1) / np.linalg.norm(C1 - B1) / np.linalg.norm(B1 - A1))
        angle9 = np.arccos(np.dot(F1 - E1, E1 - D1) / np.linalg.norm(F1 - E1) / np.linalg.norm(E1 - D1))
        angle10 = np.arccos(np.dot(C1 - B1, norm_vec1) / np.linalg.norm(C1 - B1))
        angle11 = np.arccos(np.dot(F1 - E1, norm_vec2) / np.linalg.norm(F1 - E1))
        res_vec1 = (C1 - B1) - np.dot(C1 - B1, B1 - A1) / (np.linalg.norm(B1 - A1) ** 2) * (B1 - A1)
        res_vec2 = (F1 - E1) - np.dot(F1 - E1, E1 - D1) / (np.linalg.norm(E1 - D1) ** 2) * (E1 - D1)
        angle12 = np.arccos(np.dot(res_vec1, norm_vec) / np.linalg.norm(res_vec1))
        angle13 = np.arccos(np.dot(res_vec2, norm_vec) / np.linalg.norm(res_vec2))

        check1 = (np.rad2deg(angle1) < 10 and np.rad2deg(angle1) >= 0) or (np.rad2deg(angle1) <= 180 and np.rad2deg(angle1) >= 170)
        check2 = np.rad2deg(angle2) < 110 and np.rad2deg(angle2) > 70
        check3 = np.rad2deg(angle3) < 110 and np.rad2deg(angle3) > 70
        check4 = np.rad2deg(angle4) < 150 and np.rad2deg(angle4) >= 0
        check5 = np.rad2deg(angle5) < 150 and np.rad2deg(angle5) >= 0
        check6 = np.rad2deg(angle6) < 150 and np.rad2deg(angle6) >= 0
        check7 = np.rad2deg(angle7) < 150 and np.rad2deg(angle7) >= 0
        check8 = np.rad2deg(angle8) < 150 and np.rad2deg(angle8) >= 0
        check9 = np.rad2deg(angle9) < 150 and np.rad2deg(angle9) >= 0
        check10 = np.rad2deg(angle10) < 110 and np.rad2deg(angle10) >= 0
        check11 = np.rad2deg(angle11) < 110 and np.rad2deg(angle11) >= 0
        check12 = np.rad2deg(angle12) <= 90 and np.rad2deg(angle12) >= 0
        check13 = np.rad2deg(angle13) <= 90 and np.rad2deg(angle13) >= 0

        return check1 and check2 and check3 and check4 and check5 and check6 and check7 and check8 and check9 and check10 and check11 and check12 and check13, sum([check1, check2, check3, check4, check5, check6, check7, check8, check9, check10, check11, check12, check13])


def lowercheck(O1, O2, A1, B1, C1, D1, E1, F1, A2, B2, C2, D2, E2, F2):
        O1, O2, A1, B1, C1, D1, E1, F1, A2, B2, C2, D2, E2, F2 = np.real(O1),np.real(O2),np.real(A1),np.real(B1),np.real(C1),np.real(D1),np.real(E1),np.real(F1),np.real(A2),np.real(B2),np.real(C2),np.real(D2),np.real(E2),np.real(F2)
        norm_vec = np.cross(D2 - O1, A2 - O1)
        norm_vec = norm_vec / np.linalg.norm(norm_vec)
        norm_vec1 = np.cross(B2 - O2, A2 - O2)
        norm_vec1 = norm_vec1 / np.linalg.norm(norm_vec1)
        norm_vec2 = np.cross(D2 - O2, E2 - O2)
        norm_vec2 = norm_vec2 / np.linalg.norm(norm_vec2)

        angle1 = np.arccos(np.dot(D2 - O2, O2 - A2) / np.linalg.norm(D2 - O2) / np.linalg.norm(O2 - A2))
        angle2 = np.arccos(np.dot(A2 - O2, norm_vec) / np.linalg.norm(A2 - O2))
        angle3 = np.arccos(np.dot(D2 - O2, norm_vec) / np.linalg.norm(D2 - O2))
        angle4 = np.arccos(np.dot(B2 - A2, norm_vec) / np.linalg.norm(B2 - A2))
        angle5 = np.arccos(np.dot(E2 - D2, norm_vec) / np.linalg.norm(E2 - D2))
        angle6 = np.arccos(np.dot(A2 - O2, B2 - A2) / np.linalg.norm(A2 - O2) / np.linalg.norm(B2 - A2))
        angle7 = np.arccos(np.dot(D2 - O2, E2 - D2) / np.linalg.norm(D2 - O2) / np.linalg.norm(E2 - D2))
        angle8 = np.arccos(np.dot(C2 - B2, B2 - A2) / np.linalg.norm(C2 - B2) / np.linalg.norm(B2 - A2))
        angle9 = np.arccos(np.dot(F2 - E2, E2 - D2) / np.linalg.norm(F2 - E2) / np.linalg.norm(E2 - D2))
        angle10 = np.arccos(np.dot(C2 - B2, norm_vec1) / np.linalg.norm(C2 - B2))
        angle11 = np.arccos(np.dot(F2 - E2, norm_vec2) / np.linalg.norm(F2 - E2))

        check1 = (np.rad2deg(angle1) < 10 and np.rad2deg(angle1) >= 0) or (np.rad2deg(angle1) <= 180 and np.rad2deg(angle1) >= 170)
        check2 = np.rad2deg(angle2) < 110 and np.rad2deg(angle2) > 70
        check3 = np.rad2deg(angle3) < 110 and np.rad2deg(angle3) > 70
        check4 = np.rad2deg(angle4) < 120 and np.rad2deg(angle4) >= 0
        check5 = np.rad2deg(angle5) < 120 and np.rad2deg(angle5) >= 0
        check6 = np.rad2deg(angle6) < 150 and np.rad2deg(angle6) > 30
        check7 = np.rad2deg(angle7) < 150 and np.rad2deg(angle7) > 30
        check8 = np.rad2deg(angle8) < 90 and np.rad2deg(angle8) >= 0
        check9 = np.rad2deg(angle9) < 90 and np.rad2deg(angle9) >= 0
        check10 = np.rad2deg(angle10) < 180 and np.rad2deg(angle10) >= 90
        check11 = np.rad2deg(angle11) < 180 and np.rad2deg(angle11) >= 90
        check12 = np.dot(norm_vec, B2 - A2) * np.dot(norm_vec, E2 - D2) <= 0

        return check1 and check2 and check3 and check4 and check5 and check6 and check7 and check8 and check9 and check10 and check11 and check12, sum([check1,check2,check3,check4,check5,check6,check7,check8,check9,check10,check11,check12])

        # return check1 and check2 and check3 and check4 and check5 and check6 and check7 and check8 and check9 and check10


def crosscheck(O1, O2, A1, B1, C1, D1, E1, F1, A2, B2, C2, D2, E2, F2):
        O1, O2, A1, B1, C1, D1, E1, F1, A2, B2, C2, D2, E2, F2 = np.real(O1),np.real(O2),np.real(A1),np.real(B1),np.real(C1),np.real(D1),np.real(E1),np.real(F1),np.real(A2),np.real(B2),np.real(C2),np.real(D2),np.real(E2),np.real(F2)
        norm_vec1 = -np.cross(D2 - O1, A2 - O1)
        norm_vec1 = norm_vec1 / np.linalg.norm(norm_vec1)
        norm_vec2 = np.cross(D1 - O2, A1 - O2)
        norm_vec2 = norm_vec2 / np.linalg.norm(norm_vec2)

        angle = np.arccos(np.dot(norm_vec1, norm_vec2))
        check1 = (np.rad2deg(angle) < 20 and np.rad2deg(angle) >= 0) or (np.rad2deg(angle) <= 180 and np.rad2deg(angle) >= 160)

        return check1, sum([check1])


def balance(coord_3d):  # k*14*3
        index = np.argmax(
            np.absolute(np.sum(coord_3d[:, :, -1], axis=1) - np.ones(coord_3d.shape[0]) * 14 * coord_3d[0, 0, -1]), axis=0)

        return coord_3d[index]