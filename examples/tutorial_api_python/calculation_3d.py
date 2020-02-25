@staticmethod
def get_3d_angles(keypoints_data, t_log, index_pair=angle_index_pair): # n*25*3
    keypoints_loc = np.unique(index_pair.flatten())
    # jump=[i for i in range(1, np.diff(t_log).shape[0] - 1) if
    #  np.diff(t_log)[i] > 3 * np.diff(t_log)[i - 1] and np.diff(t_log)[i] > 3 * np.diff(t_log)[i + 1]]
    # t_log = np.delete(t_log,jump)
    # keypoints_data = np.delete(keypoints_data,jump,axis=0)

    valid_mask = np.all(~np.any(keypoints_data[:, keypoints_loc, :] == 0, axis=1), axis=1)
    keypoints_data = keypoints_data[valid_mask, :, :] # n*14*3
    t_log = t_log[valid_mask]
    keypoints_data = from_2d_to_3d(keypoints_data,Person().ratio,Person().torso_length) # n*14*3
    angle_3d = np.zeros((len(t_log), 0))
    for i in range(index_pair.shape[0]):
        vector1 = keypoints_data[:, index_pair[i][1], 0:3] - keypoints_data[:, index_pair[i][0], 0:3]
        vector2 = keypoints_data[:, index_pair[i][3], 0:3] - keypoints_data[:, index_pair[i][2], 0:3]
        angle = np.arccos(np.sum(vector1 * vector2, axis=1) / np.sqrt(
            np.sum(vector1 * vector1, axis=1) * np.sum(vector2 * vector2, axis=1)))
        # angle = angle_all[~np.isnan(angle_all)]
        angle_3d = np.concatenate((angle_3d, angle.reshape(-1, 1)), axis=1)
    angle_3d_dot = np.gradient(angle_3d, t_log, axis=0)
    angle_3d_ddot = np.gradient(angle_3d_dot, t_log, axis=0)

    return np.concatenate((angle_3d, angle_3d_dot, angle_3d_ddot), axis=1)

# def from_2d_to_3d(keypoints_data,Person().ratio,Person().torso_length,Person().keypoints_loc):
def from_2d_to_3d(keypoints_data):
    lambda = Person().focal_length
    list_3d = np.zeros((0,14,3))
    # for count, coord_2d in enumerate(keypoints_data[:,keypoints_loc,:][:,:,0:2]): # coord_2d: 14*2
    for coord_2d in keypoints_data[:, keypoints_loc, :][:, :, 0:2]:  # coord_2d: 14*2
        lt = numpy.linalg.norm(coord_2d[Joints(mid_hip).value]-coord_2d[Joints(neck).value])
        k0 = Person().torso_length/lt
        index_leftarm = np.array([Joints(neck).value,Joints(l_shoulder).value,Joints(l_elbow).value,Joints(l_wrist).value])
        index_rightarm = np.array([Joints(neck).value, Joints(r_shoulder).value, Joints(r_elbow).value, Joints(r_wrist).value])
        index_leftleg = np.array([Joints(mid_hip).value, Joints(l_hip).value, Joints(l_knee).value, Joints(l_ankle).value])
        index_rightleg = np.array([Joints(mid_hip).value, Joints(r_hip).value, Joints(r_knee).value, Joints(r_ankle).value])
        index_body = np.vstack((index_leftarm,index_rightarm,index_leftleg,index_rightleg))
        # k_list = np.zeros((0,4))
        for index in range(index_body.shape[0]):
            index_part = index_body[index,:]
            k_old = np.array([k0])
            for i in range(index_part)-1:
                startpoint = coord_2d[index_part[i],:]
                endpoint = coord_2d[index_part[i+1],:]
                x1 = startpoint[0]
                y1 = startpoint[1]
                x2 = endpoint[0]
                y2 = endpoint[1]
                if len(k_old.shape)>1:
                    k_temp = np.zeros((0,k_old.shape[1]+1))
                    for j in range(k_old.shape[0]):
                        k_new = np.roots([x2 ^ 2 + y2 ^ 2 + lambda ^ 2, -2 * k_old[j,0] * (x1 * x2 + y1 * y2 + lambda ^ 2), k_old[j,0] ^ 2 * (x1 ^ 2 + y1 ^ 2 + lambda ^ 2) - L ^ 2])
                        k_temp = np.concatenate((k_temp,np.hstack((k_new.reshape[-1,1],np.repeat(k_old[j,:],k_new.shape[0],axis=1)))),axis=0)
                    k_old = k_temp
                else:
                    k_new = np.roots([x2 ^ 2 + y2 ^ 2 + lambda ^ 2, -2 * k_old[j,0] * (x1 * x2 + y1 * y2 + lambda ^ 2), k_old[j,0] ^ 2 * (x1 ^ 2 + y1 ^ 2 + lambda ^ 2) - L ^ 2])
                    k_old = np.hstack((k_new.reshape[-1,1],np.repeat(k_old[j,:],k_new.shape[0],axis=1))) # k_old: (8 or 4)*4

            if index == 0:
                k_list = k_old
            else:
                k_list = np.array([k_list,k_old]) # k_list: 4* [(8 or 4)*4]
        # check the 3d coords corresponding to these k values, in the order of 4 body parts
        for i in range(k_list.shape[0]):
            index_part=index_body[i,:]
            coord = coord_2d[index_part,:] # 4*2
            coord_expand = np.hstack((coord,lambda*np.ones(4,1))) # 4*3
            coord_part_3d = np.zeros((0,4,3))
            for j in range(k_list[i].shape[0]):
                coord_expand_3d = k_list[i][j].reshape(-1,1)*coord_expand # 4*3
                coord_part_3d = np.concatenate((coord_part_3d,coord_expand_3d.reshape[1,4,3]),axis=0) # (8 or 4)*4*3
            if i == 0:
                coord_3d = np.array([coord_part_3d])
            else:
                coord_3d = np.array([coord_3d,coord_part_3d]) # this coord_3d is actually the set of all possible values,
                # it has 4 dimension, 4*(8 or 4*4*3), the first 4 represents four part: (left,right) x (arm,leg)

        coord_3d = verify(coord_3d) # k*14*3
        coord_3d = balance(coord_3d) # 14*3
        list_3d = np.concatenate((list_3d, coord_3d.reshape[1, 14, 3]), axis=0)
        # if count ==0:
        #     list_3d = coord_3d
        # else:
        #     list_3d = np.array([list_3d, coord_3d])
        # coord_3d = maxprob(coord_3d)

    return list_3d    # n*14*3


@staticmethod
def verify(coord_3d):
    O1 = coord_3d[0,0,-1,:]
    O2 = coord_3d[-1,0,-1,:]
    good_list = np.zeros((0,14,3))
    # good_list = np.zeros((0,42))
    for coord_leftarm in coord_3d[0][:,0:-1,:]: # 8 or 4 *3*3
        for coord_rightarm in coord_3d[1][:,0:-1,:]:
            for coord_leftleg in coord_3d[2][:, 0:-1, :]:
                for coord_rightleg in coord_3d[3][:, 0:-1, :]:
                    C1 = coord_leftarm[0,:]
                    B1 = coord_leftarm[1,:]
                    A1 = coord_leftarm[2,:]
                    F1 = coord_rightarm[0,:]
                    E1 = coord_rightarm[1,:]
                    D1 = coord_rightarm[2,:]
                    C2 = coord_leftleg[0,:]
                    B2 = coord_leftleg[1,:]
                    A2 = coord_leftleg[2,:]
                    F2 = coord_rightleg[0,:]
                    E2 = coord_rightleg[1,:]
                    D2 = coord_rightleg[2,:]
                    if uppercheck(O1,O2,A1,B1,C1,D1,E1,F1,A2,B2,C2,D2,E2,F2) and lowercheck(O1,O2,A1,B1,C1,D1,E1,F1,A2,B2,C2,D2,E2,F2) and crosscheck(O1,O2,A1,B1,C1,D1,E1,F1,A2,B2,C2,D2,E2,F2):
                        good_3d=np.vstack((O1,O2,A1,B1,C1,D1,E1,F1,A2,B2,C2,D2,E2,F2)).T  # 14*3
                        good_list=np.concatenate((good_list,good_3d.reshape[1,14,3]),axis=0) # k*14*3, k is the number of good combinitions
                        # good_list=np.concatenate((good_list,good_3d),axis=0) # x*42, x is the number of good combinitions
                        # good_list=good_list.reshape[-1,14,3]
    return good_list # k*14*3


def uppercheck(O1,O2,A1,B1,C1,D1,E1,F1,A2,B2,C2,D2,E2,F2):
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

    check1 = np.rad2deg(angle1) < 10 and np.rad2deg(angle1) >= 0
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

    return check1 and check2 and check3 and check4 and check5 and check6 and check7 and check8 and check9 and check10 and check11 and check12 and check13


def lowercheck(O1,O2,A1,B1,C1,D1,E1,F1,A2,B2,C2,D2,E2,F2):
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

    check1 = np.rad2deg(angle1) < 10 and np.rad2deg(angle1) >= 0
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

    return check1 and check2 and check3 and check4 and check5 and check6 and check7 and check8 and check9 and check10 and check11 and check12


def crosscheck(O1,O2,A1,B1,C1,D1,E1,F1,A2,B2,C2,D2,E2,F2):
    norm_vec1 = -np.cross(D2 - O1, A2 - O1)
    norm_vec1 = norm_vec1 / np.linalg.norm(norm_vec1)
    norm_vec2 = np.cross(D1 - O2, A1 - O2)
    norm_vec2 = norm_vec2 / np.linalg.norm(norm_vec2)

    angle = np.arccos(np.dot(norm_vec1, norm_vec2))
    check1 = np.rad2deg(angle) < 20 & & np.rad2deg(angle) >= 0

    return check1




def balance(coord_3d): # k*14*3
    index = np.argmax(np.absolute(np.sum(coord_3d[:,:,-1],axis=1)-np.ones(coord_3d.shape[0])*14*coord_3d[0,0,-1]),axis=0)

    return coord_3d[index]