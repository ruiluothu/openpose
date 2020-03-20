import os
import glob
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from people_feature_generation import Person
from sklearn import svm
from sklearn.model_selection import cross_val_score


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


train_path = 'training'
actions = ['walking','running','other']
labels = np.array([0,1,2])

comb = [np.arange(8), np.arange(8,16), np.arange(16,24), np.arange(16), np.arange(8,24), np.arange(24)]
scores = np.zeros((len(comb), 10))
for j in range(len(comb)):
    X = np.zeros((0,comb[j].shape[0]))
    y = np.zeros(0)

    for index, action in enumerate(actions):
        loc = os.path.join(train_path, action)
        files = glob.glob(os.path.join(loc, '*.json'))
        features = np.zeros((0, comb[j].shape[0]))
        for file in files:
            data, t = _reading_json_file(file)
            feature = Person().get_2d_angles(data, t) # n*24
        # all possible combination of the training data
            feature = feature[:, comb[j]]
            features = np.concatenate((features,feature), axis=0)
        # X = np.concatenate((X, feature), axis=0)  # n*16
        # y = np.concatenate((y, labels[index] * np.ones(feature.shape[0])), axis=0)  # n
        features = -1 + 2*(features - np.min(features, axis=0))/(np.max(features, axis=0) - np.min(features, axis=0))
        X = np.concatenate((X, features), axis=0) # n*16
        y = np.concatenate((y, labels[index]*np.ones(features.shape[0])), axis=0) # n
# X = -1 + 2*(X - np.min(X, axis=0))/(np.max(X, axis=0) - np.min(X, axis=0))

    clf1 = svm.SVC(kernel='rbf', gamma='scale')
    scores[j] = cross_val_score(clf1, X, y, cv=10)

    print(np.mean(scores[j]))
# np.savetxt('result.txt', scores, fmt='%f')

##################### plot the feature value for each class ####################
# sample1 = X[[0,1,2,3],:]
# sample2 = X[[-1,-2,-3,-4],:]
# samplelabel1 = y[[0,1,2,3]]
# samplelabel2 = y[[-1,-2,-3,-4]]
# plt.plot(np.arange(X.shape[1]),sample1.T, Linestyle='-.')
# plt.plot(np.arange(X.shape[1]),sample2.T, Linestyle='-')
# plt.xlabel('No. # of features')
# plt.ylabel('Values')
# plt.legend(['walk','walk','walk','walk','run','run','run','run'])  # np.repeat(np.array(['walk']),3).tolist()
# plt.show()

########################### test with SVM #################################
# from sklearn import svm
# clf1 = svm.SVC(kernel='rbf', gamma='scale')
# from sklearn.model_selection import cross_val_score
# scores1 = cross_val_score(clf1, X, y, cv=10)
# np.savetxt('test2.txt', scores1, fmt='%f')
# print(scores1)

# scores1 = np.loadtxt('test3.txt', dtype=float)
# scores2 = np.loadtxt('test2.txt', dtype=float)
# scores3 = np.loadtxt('test1.txt', dtype=float)
# plt.plot(np.arange(len(scores1)), np.array(scores1),Linestyle='-')
# plt.plot(np.arange(len(scores1)), np.array(scores2),Linestyle=':')
# plt.plot(np.arange(len(scores1)), np.array(scores3),Linestyle='-.')
# plt.ylim([0.5,1.2])
# plt.title('Tenfold Validation of SVM')
# plt.legend(['Normalization in each group', 'Normalization in all data', 'No normalization'])
# plt.savefig('svmscore.jpg')

########################### try use pytorch neural networks ###########################
# import torch
# import torch.nn.functional as F
#
# xdata = torch.from_numpy(X).type(torch.FloatTensor)
# ydata = torch.from_numpy(y).type(torch.LongTensor)
#
# class Net(torch.nn.Module):
#     def __init__(self, n_feature, n_hidden, n_hidden2, n_hidden3, n_output):
#         super(Net, self).__init__()
#         self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
#         self.hidden2 = torch.nn.Linear(n_hidden, n_hidden2)
#         self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)
#         self.out = torch.nn.Linear(n_hidden3, n_output)   # output layer
#
#     def forward(self, x):
#         x = F.relu(self.hidden(x))      # activation function for hidden layer
#         x = F.relu(self.hidden2(x))
#         x = F.relu(self.hidden3(x))
#         x = self.out(x)
#         return x
#
#
# net = Net(n_feature=16, n_hidden=100, n_hidden2=50, n_hidden3=20, n_output=len(actions))     # define the network
# print(net)  # net architecture
#
# optimizer = torch.optim.SGD(net.parameters(), lr=0.25)
# loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted
#
# plt.ion()   # something about plotting
#
# total_accuracy=np.zeros(100, dtype=float)
# for t in range(100):             # training epochs
#     out = net(xdata)                 # input x and predict based on x
#     loss = loss_func(out, ydata)     # must be (1. nn output, 2. target), the target label is NOT one-hotted
#
#     optimizer.zero_grad()   # clear gradients for next train
#     loss.backward()         # backpropagation, compute gradients
#     optimizer.step()        # apply gradients
#
#
#     if t % 2 == 0:
#         # plot and show learning process
#         # plt.cla()
#         prediction = torch.max(out, 1)[1]
#         pred_y = prediction.data.numpy()
#         target_y = ydata.data.numpy()
#         # plt.scatter(xdata.data.numpy()[:, 0], xdata.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
#         accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
#         total_accuracy[[t,t+1]]=np.repeat(accuracy,2)
#         # plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
#         # plt.pause(0.05)
#
# print(total_accuracy)
# plt.ioff()
# plt.show()