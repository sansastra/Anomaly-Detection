import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # for removing unnecessary warnings
from absl import logging
logging._warn_preinit_stderr = 0
logging.warning('...')
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import itertools
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from load_data_last_seen import load_normal_data, load_ano_data, load_all_data, load_test_data
#from keras.models import Sequential
#from keras.layers import Dense
import pickle
import joblib
from PIL import Image
import time

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45) # ,
    plt.yticks(tick_marks, classes) # this line scales y-axis so labels and numbers does not look good in CM

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


t11, f12, f13, f21, t22, f23, f31, f32, t33 = 0, 0, 0, 0, 0, 0, 0, 0, 0
acc = 0

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
features = ['x', 'y', 'cog', 'sog']
dim = len(features)
timesteps = 60  # number of sequences per batch
CLASSES = 2




data_normal, Y_data = load_all_data(timesteps, dim, features,CLASSES)
#if len(Y_data)>1:
#data_normal, Y_data = load_all_data_new(timesteps,dim, features,CLASSES)

nr_iter = 10
train_time, test_time =0, 0
    #print('dataset is labeled')
for val in range(nr_iter):
    np.random.seed(val)
    data_train, data_test, target_train, target_test = train_test_split(data_normal, Y_data, test_size=0.40)

    #cross_validate_data = data_test
    # # data pre-processing
    # scaler = StandardScaler() #MinMaxScaler() #
    # # Fit only to the training data
    # scaler.fit(data_train)
    # # apply the transformations to the data:
    # data_train = scaler.transform(data_train)
    #
    # #scaler.fit(data_test) #test on real data without scaling
    # data_test = scaler.transform(data_test)
    start_time = time.time()
    ############ ANN model #######################
    # Create tree classifier
    clf = MLPClassifier(solver='adam', alpha=1e-4,learning_rate='adaptive',
                        hidden_layer_sizes=(100,), random_state=1, verbose=1,
                        tol=1e-5)  # alpha => regularization parameter; random_state => RNG seed
    clf.out_activation_ = 'softmax'
    # Fit the data
    clf.fit(data_train, target_train)
    train_time += (time.time() - start_time)*1000/data_train.shape[0]


    start_time = time.time()
    # Predict the response

    pred = clf.predict_proba(data_test)
    test_time += (time.time() - start_time)*1000/data_test.shape[0]
    #pred2 = clf.predict_proba(data_test[0:1,:])


    acc += accuracy_score(target_test.argmax(axis=1), pred.argmax(axis=1))

    # print('total number of validated power outage samples is ', sum(target_test==[0, 1, 0])[1])
    # print('total number of validated anomaly samples is ', sum(target_test==[0, 0, 1])[2])
    # print('total number of validated normal samples is ', sum(target_test==[1, 0, 0])[0])

    #cm = confusion_matrix(target_test.argmax(axis=1), pred.argmax(axis=1))
    #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #print('confusion matrix : \n',cm)
    #cnf_matrix = confusion_matrix(target_test.argmax(axis=1), pred.argmax(axis=1))
    #print('classification_report : \n', classification_report(y_test,pred))

    # aa=target_test.argmax(axis=1)
    # bb = pred.argmax(axis=1)
    # x =  np.where((aa != bb))[0]

    #print('y_test :',y_test)
    #print('predic :',pred.tolist())
    if CLASSES ==2:
        r11, r12, r21, r22 = confusion_matrix(target_test.argmax(axis=1), pred.argmax(axis=1)).ravel()
        t11 += r11
        f12 += r12
        f21 += r21
        t22 += r22

    else:

        r11, r12, r13, r21, r22, r23, r31, r32, r33 = confusion_matrix(target_test.argmax(axis=1), pred.argmax(axis=1)).ravel()
        t11 += r11
        f12 += r12
        f13 += r13
        f21 += r21
        t22 += r22
        f23 += r23
        f31 += r31
        f32 += r32
        t33 += r33

    print("Completed run ", val)
if CLASSES == 2:
    print(">>>>>>>>>>>>>>>>>>>>>>>>> \n accuracy : ", acc / nr_iter)
    cm = np.reshape(np.array([t11, f12, f21, t22]), (2, 2))
    #print(cm)

    #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    # plot_confusion_matrix(cm, ['normal', 'anomaly'], normalize=False, title='Confusion matrix')
    # plt.show()

    # save the model to disk
    filename = 'ANN_model_90_2_classes.sav'
    joblib.dump(clf, filename)
else:
    print(">>>>>>>>>>>>>>>>>>>>>>>>> \n accuracy : ", acc/nr_iter)
    cm = np.reshape(np.array([t11, f12, f13, f21, t22, f23, f31, f32, t33]),(3,3))
    #print(cm)

    #cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    #plot_confusion_matrix(cm,['normal','power outage', 'anomaly'], normalize=False, title='Confusion matrix')
    #plt.show()

    # save the model to disk
    filename = 'ANN_model_90_3_classes.sav'
    joblib.dump(clf, filename)

print('train time per 1000 samples= ', train_time/nr_iter)
print('test time per 1000 samples= ', test_time/nr_iter)

#####################
#model = Sequential()
#model.add(Dense(12, input_dim=data_train.shape[1], init='uniform', activation='relu'))
#model.add(Dense(10, init='uniform', activation='relu'))
#model.add(Dense(3, init='uniform', activation='softmax'))
#model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# Train model
#history = model.fit(data_train, target_train, nb_epoch=10, batch_size=50, verbose=0)
# Print Accuracy
#scores = model.evaluate(data_test, target_test)
#pred = model.predict(data_test)

#cm = confusion_matrix(target_test.argmax(axis=1), pred.argmax(axis=1))
#cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]




#print(classification_report(target_test, pred))
        # fig, ax = plt.subplots(figsize=(8, 7))
        # im = Image.open('world_11-2_54_12-4_55.PNG')  # in degrees and minutes
        # ax.imshow(im, extent=(11.3333, 12.6666, 54.0, 55), aspect='auto')
        # ax.plot(data_normal[:, 0], data_normal[:, 1], '.k', label='Trajectory of a vessel')
        # ax.plot(cross_validate_data[:, 0], cross_validate_data[:, 1], '.b', label='Validated data of a vessel')
        # plt.pause(0.01)
        #
        # xy = {}
        # # normal=[1 0 0], power_outage= [0, 1, 0], AIS on_off = [0 0 1]
        # # ano_loc = np.where(pred.argmax(axis=1) == 1)[0]
        # ano_loc = np.where(((pred[:, 1] > 0.9) * (pred[:, 0] < 0.1) * (pred[:, 2] < 0.1)))[0]
        # # ax.plot(data_to_test.iloc[ano_loc]['x'], data_to_test.iloc[ano_loc]['y'], 'b>', markersize=12, label='power outage')
        # for i in np.array(ano_loc):
        #     ax.plot(cross_validate_data[i, 0], cross_validate_data[i, 1], 'b>', markersize=12, label='Power outage')
        # ano_loc = np.where(((pred[:, 2] > 0.9) * (pred[:, 0] < 0.1) * (pred[:, 1] < 0.1)))[0]
        # # ax.plot(data_to_test.iloc[ano_loc]['x'], data_to_test.iloc[ano_loc]['y'], 'rx', markersize=8, label='AIS anomaly')
        # for i in np.array(ano_loc):
        #     ax.plot(cross_validate_data[i, 0], cross_validate_data[i, 1], 'rx', markersize=8, label='AIS anomaly')
        # plt.xlabel('Longitude')
        # plt.ylabel('Latitude')
        # plt.title('AIS on-off switching anomaly detection')
        # ax.legend()
        # plt.show()
        # plt.pause(0.01)
