import numpy as np
import pandas as pd
from tensorflow import keras
from ENUtransform import WGS84toENU, ENUtoWGS84
from PythonCode.Trajectory_Prediction.process import scale_data, reshape_data, get_inverse_transform

tREF = {"lon": 12.114733,
            "lat": 54.145409,
            "ECEF": np.array([[3660725], [785776], [514624]])
        }
data_features = ['x', 'y', 'cog', 'sog']
data_dim = len(data_features)

enu_features = ['x', 'y', "z", 'cog', 'sog']
enu_dim = len(enu_features)
INPUT_LEN = 10  # same as timesteps
model_name = 'Seq2Seq_model_ENU.h5' # 'Seq2Seq_model_ENU_167.h5'
model = keras.models.load_model('/home/sing_sd/Desktop/anomaly_detection/PythonCode/Trajectory_Prediction/'+ model_name)

def load_data_trajectory(filename):
    path = '/home/sing_sd/Desktop/anomaly_detection/PythonCode/KF/'

    filename1 = path + filename # "Track167_interpolated_1min.csv"
    # filename4 = path + "Track167_EKF.csv"

    try:
       return np.array(pd.read_csv(filename1))
    except IOError:
        print("Error: File does not appear to exist for track ")
        return None


def convert2ENU(lon, lat):
    return WGS84toENU(lon, lat, tREF, h=0.)


def convert2Degree(zENU):
    return ENUtoWGS84(np.array(zENU), tREF)

def data_preparation(xhat_past, data):
    in_clm_len = INPUT_LEN*enu_dim
    overall_data = np.full(shape=(1, in_clm_len), fill_value=np.nan)
    overall_data[0, 0:in_clm_len:enu_dim] = xhat_past[:, 0].ravel()
    overall_data[0, 1:in_clm_len:enu_dim] = xhat_past[:, 1].ravel()
    overall_data[0, 2:in_clm_len:enu_dim] = xhat_past[:, 2].ravel()
    overall_data[0, 3:in_clm_len:enu_dim] = np.transpose(data[:, 2])
    overall_data[0, 4:in_clm_len:enu_dim] = np.transpose(data[:, 3])

    return overall_data


def predict_data(xhat_past, data):
    test_data = data_preparation(np.array(xhat_past), data)
    X_test = test_data.reshape(1, INPUT_LEN, enu_dim)
    X_test[0] = scale_data(X_test[0])
    test_predict = model.predict(X_test)

    # invert predictions
    test_predict = get_inverse_transform(test_predict[0])
    test_predict.shape = (1, INPUT_LEN * enu_dim)
    # convert lon, lat from ENU to degrees
    return test_predict[0, 0], test_predict[0, 1], test_predict[0, 2], test_predict[0, 3], test_predict[0, 4]
