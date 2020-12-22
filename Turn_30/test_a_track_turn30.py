import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # for removing unnecessary warnings
from absl import logging
logging._warn_preinit_stderr = 0
logging.warning('...')
from tensorflow import keras
import numpy as np
from load_data_30_turn import load_test_data, load_test_data_time
import matplotlib.pyplot as plt
import joblib


import pandas as pd
import numpy as np
from scipy import interpolate


########### check a whole track ##############

#features = ['x', 'y', 'cog', 'sog']
features = ['cog']
dim = len(features)
timesteps = 60


track_to_check = 167  # 43, 167, 202
model_name = 'ANN_30_turn_rostock1_new.h5' # lstm, Seq2Seq, Seq_at
model = joblib.load(model_name)


plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(figsize=(8, 8))


def test_track(features, dim, track_to_check):
    #test_data, target_data = load_test_data(timesteps, dim, track_to_check)
    original_data, interpolated_data, test_data, target_data = load_test_data_time(timesteps, dim, features, track_to_check)
    # original_data.plot(kind='scatter', x=2, y=3, color='black')
    ax.plot(original_data.iloc[:, 2], original_data.iloc[:, 3], '.k', label='Original trajectory')
    test_predict = model.predict_proba(test_data)
    #plt.show()
    plt.pause(0.0001)
    ind_ano = np.where(((test_predict[:, 1] > 0.99) * (test_predict[:, 0] < 0.01)))[0]
# range(dim, timesteps * dim, dim)

    ax.plot(interpolated_data[ind_ano, 0], interpolated_data[ind_ano, 1], 'xb',  label='Turn detected')

    ind_ano_actual = np.where(((target_data[:, 1] > 0.9) * (target_data[:, 0] < 0.1)))[0]
    ax.plot(interpolated_data[ind_ano_actual, 0], interpolated_data[ind_ano_actual, 1], '.r', label='Actual turn')
    plt.pause(0.0001)
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees)')
    # plt.title('AIS on-off switching anomaly detection')
    ax.legend()
    plt.draw()
 #   plt.savefig('turn_pred.png')


test_track(features, dim, track_to_check)
plt.pause(0.001)

# testScore = math.sqrt(mean_squared_error(target_test[:len(test_broken_predict1),0], test_broken_predict1[:,0])) + \
#             math.sqrt(mean_squared_error(target_test[:len(test_broken_predict1),1], test_broken_predict1[:,1]))
# print('Anomaly test Score: %.2f RMSE' % (testScore))

################################################################################