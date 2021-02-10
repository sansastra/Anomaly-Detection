import numpy as np
import pandas as pd
from math import sin, cos, sqrt, atan2, radians
import matplotlib.pyplot as plt
import os
import h5py
import pickle
from PIL import Image
from datetime import date, time, datetime
from itertools import groupby
from operator import itemgetter
import math
from os.path import join
interactive = True
headers=['index1', 'ship_nr','id','repeat_indicator','mmsi','nav_status','rot_over_range','rot','sog','position_accuracy','x','y','cog','true_heading',
         'timestamp','special_manoeuvre','spare','raim','sync_state','slot_timeout','slot_offset', 'abs_time', 'date', 'time']


ROSTOCK = (12.114733, 54.145409)
Dummy_Nr = -1
CLASSES = 2
EXTRA_FEATURES = 0 # one for distance, and another for time to last seen
SAMPLING_TIME = 3 # seconds
MINSOG = 7
MINCOGCHANGE = 30

nr_of_vessels = 10 # -1


def load_all_data(timesteps, dim, features, CLASSES): # without taking time into account
    ############## generate data from pickle to train ANN ##
    np.random.seed(10)
    column_len = dim * timesteps
    cog_index = features.index("cog") # if features is changed then cog position might change so later it is used correctly
    with open('/home/sing_sd/Desktop/anomaly_detection/PythonCode/Resources/ais_data_rostock.csv', 'rb') as f: #
        data = pd.read_csv(f)

    # plt.rcParams.update({'font.size': 16})
    # fig, ax = plt.subplots(figsize=(8, 7))
    # im = Image.open('world_11_54_12-4_55.PNG')  # in degrees and minutes
    # ax.imshow(im, extent=(11, 12.6666, 54.0, 55), aspect='auto')
    # ax.plot(data['x'], data['y'], 'w.',markersize=2, label='Trajectories of vessels')
    # boundary_x, boundary_y = get_po_boundary()
    # ax.plot(boundary_x, boundary_y, 'k-', markersize=8, label='AIS transmission reach')
    # ax.plot(ROSTOCK[0], ROSTOCK[1], 'ko', markersize=12, label='Rostock location')
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # # plt.title('AIS on-off switching anomaly detection')
    # ax.legend()
    # plt.show()
    # plt.pause(0.1)

    # with open('data_len_track.pkl', 'rb') as f:
    #     data_all_tracks = pickle.load(f)
    # vessel_nr = 18 #28
    data1 = data.mmsi.unique()

    overall_data = np.full(shape=(data.shape[0], column_len), fill_value=np.nan)

    startIndex = 0
    for mmsi in data1[:nr_of_vessels]: #[vessel_nr:vessel_nr+1]: #

        decoded_mmsi = data[data['mmsi'] == mmsi]
        decoded_mmsi = decoded_mmsi.reset_index(drop=True)
        decoded_mmsi = decoded_mmsi[decoded_mmsi['sog'] > MINSOG]

        # decoded_mmsi.plot(kind='scatter', x=1, y=2, color='red')


        if decoded_mmsi.shape[0] > timesteps:

            data_per_track = decoded_mmsi.shape[0]

            overall_data[startIndex:startIndex + data_per_track, 0:dim] = decoded_mmsi[features] # decoded_mmsi.iloc[:, 3:dim+3]


            # shift from top and put on remaining columns
            for clm_nr in range(1, timesteps):
                overall_data[startIndex : startIndex + data_per_track  - clm_nr,
                clm_nr * dim:(clm_nr + 1) * dim] = overall_data[startIndex + 1 : startIndex + data_per_track - clm_nr + 1, (clm_nr - 1) * dim:clm_nr * dim]
            overall_data[startIndex + data_per_track -timesteps +1 : startIndex + data_per_track,:] = np.nan
            startIndex += data_per_track - timesteps +1


    overall_data = overall_data[np.where(overall_data[:, 0 :1] >= 0)[0]]
    # compute number of missing zeros after the first meassage in each row
    clm_ind = range(cog_index, timesteps*dim, dim) # cog_index is the column number in features array
    max_cog_val = np.nanmax(overall_data[:, clm_ind], axis=1)
    min_cog_val = np.nanmin(overall_data[:, clm_ind], axis=1)
    max_cog_val = max_cog_val.T
    min_cog_val = min_cog_val.T
    # find anomaly samples
    ind_row1 = np.where(((abs(max_cog_val - min_cog_val) > MINCOGCHANGE) ))[0]
    # ind_ano = ind_row1[np.where((abs(360. - max_cog_val[ind_row1] - min_cog_val[ind_row1]) > MINCOGCHANGE))[0]]
    # find false anomalies
    ind_ano1 = ind_row1[(max_cog_val[ind_row1] > 360 - MINCOGCHANGE) & (min_cog_val[ind_row1] < MINCOGCHANGE)]
    false_ano = np.array([])

    for j in ind_ano1:
        aa = np.where((overall_data[j, :] - MINCOGCHANGE <= 0))[0]
        bb = np.where((overall_data[j, :] - MINCOGCHANGE > 0))[0]
        max_cog_val_j = np.nanmax(overall_data[j, aa])  # finds max cog in 360+ data = angle rightside of North
        min_cog_val_j = abs(360-np.nanmin(overall_data[j, bb]))  # finds min cog in 360- data and then angle leftside
        if max_cog_val_j + min_cog_val_j <= 30:
            false_ano = np.append(false_ano, np.where((ind_row1 == j))[0])

    # delete false anomalies
    ind_ano = np.delete(ind_row1, false_ano)

    ind_row_normal = np.delete(np.where(overall_data[:, 0:1] >= 0)[0], ind_ano)
    delete_normal = np.random.choice(np.arange(len(ind_row_normal)), overall_data.shape[0] - 2 * len(ind_ano),
                                     replace=False)
    overall_data[ind_row_normal[delete_normal], 0] = np.nan
    # overall_data[ind_row_normal[0: overall_data.shape[0]-2*len(ind_ano)], 0] = np.nan

    # assign target values
    Y_data = np.zeros((overall_data.shape[0], CLASSES))
    Y_data[:, 0] = 1
    Y_data[ind_ano] = [0, 1]

    where_are_NaNs = np.isnan(overall_data[:, 0])
    Y_data[where_are_NaNs, 0] = np.nan
    overall_data = overall_data[~where_are_NaNs]
    Y_data = Y_data[~where_are_NaNs]

    print('total number of normal samples is ', len(Y_data) - len(ind_ano))
    print('total number of anomaly samples is ', len(ind_ano))


    # plt.rcParams.update({'font.size': 16})
    # fig, ax = plt.subplots(figsize=(8, 7))
    # #im = Image.open('world_11_54_12-4_55.PNG')  # in degrees and minutes
    # #ax.imshow(im, extent=(7, 9, 53.0, 55), aspect='auto')
    # #ax.plot(overall_data[:,0], overall_data[:,1], 'w.',label='Trajectories of vessels')
    #
    # plt.pause(0.001)
    # for i in range(len(ind_ano)):
    #     for j in range(timesteps):
    #         ax.plot(overall_data[ind_ano[i], j*dim], overall_data[ind_ano[i], j*dim +1], 'bo', markersize=6)
    #
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.title('AIS on-off switching anomaly detection')
    # # ax.legend()
    # plt.show()
    # plt.pause(0.01)

    np.savetxt("X_data.csv", overall_data, delimiter=",")
    np.savetxt("Y_data.csv", Y_data, delimiter=",")
    # overall_data[np.isnan(overall_data)] = Dummy_Nr

    return overall_data, Y_data

def load_data(timesteps, dim, features, CLASSES): # this is with dummy number, or time
    ############## generate data from pickle to train ANN ##
    np.random.seed(10)

    column_len = dim * timesteps
    cog_index = features.index("cog")  # if features is changed then cog position might change so later it is used correctly

    with open('/home/sing_sd/Desktop/anomaly_detection/PythonCode/Resources/ais_data_rostock.csv', 'rb') as f: #
        data = pd.read_csv(f)

    data1 = data.mmsi.unique()

    total_rows_data = 0
    for mmsi in data1[:nr_of_vessels]: #[vessel_nr:vessel_nr+1]: #

        decoded_mmsi = data[data['mmsi'] == mmsi]
        decoded_mmsi = decoded_mmsi[decoded_mmsi['sog'] > MINSOG]
        decoded_mmsi = decoded_mmsi.reset_index(drop=True)
        if decoded_mmsi.shape[0] > timesteps:
            total_rows_data += int((decoded_mmsi.iloc[-1]['time']- decoded_mmsi.iloc[0]['time'])// SAMPLING_TIME + 1)

    overall_data = np.full(shape=(total_rows_data, column_len), fill_value=np.nan)

    startIndex = 0
    for mmsi in data1[:nr_of_vessels]: #[vessel_nr:vessel_nr+1]: #

        decoded_mmsi = data[data['mmsi'] == mmsi]
        #decoded_mmsi = decoded_mmsi.reset_index(drop=True)
        decoded_mmsi = decoded_mmsi[decoded_mmsi['sog'] > MINSOG]
        decoded_mmsi = decoded_mmsi.reset_index(drop=True)


        if decoded_mmsi.shape[0] > timesteps:

            data_per_track = int((np.array(decoded_mmsi.iloc[-1]['time'])- decoded_mmsi.iloc[0]['time'])// SAMPLING_TIME + 1)
            decoded_mmsi['time_0'] = decoded_mmsi.iloc[0]['time']

            decoded_mmsi['time_0'] = (decoded_mmsi['time'] - decoded_mmsi['time_0'])/SAMPLING_TIME

            temp_data = pd.DataFrame(index=range(data_per_track), columns=features, dtype=np.float)
            temp_data.iloc[np.array(decoded_mmsi['time_0'], dtype=int), 0:dim] = np.array(decoded_mmsi[features])

            # interpolate
            #temp_data = temp_data.interpolate(method='linear', columns=features, limit_direction='forward', axis=0)
            temp_data = temp_data.fillna(method="ffill")

            temp_data.loc[temp_data["cog"] > 360,"cog"] = 360
            overall_data[startIndex: startIndex+data_per_track, 0:dim] = temp_data[features]

            # shift from top and put on remaining columns
            for clm_nr in range(1, timesteps):
                overall_data[startIndex : startIndex + data_per_track  - clm_nr,
                clm_nr * dim:(clm_nr + 1) * dim] = overall_data[startIndex + 1 : startIndex + data_per_track - clm_nr + 1, (clm_nr - 1) * dim:clm_nr * dim]

            overall_data[startIndex + data_per_track -timesteps +1 : startIndex + data_per_track,:] = np.nan
            startIndex += data_per_track - timesteps + 1

    overall_data = overall_data[np.where( overall_data[:, 0:1] >= 0)[0]]

    # compute number of missing zeros after the first meassage in each row
    # overall_data = np.unique(overall_data, axis=0)
    clm_ind = range(cog_index, timesteps*dim, dim)
    max_cog_val = np.nanmax(overall_data[:, clm_ind], 1)
    min_cog_val = np.nanmin(overall_data[:, clm_ind], 1)
    max_cog_val = max_cog_val.T
    min_cog_val = min_cog_val.T

    # many rows are duplicating, np.unique does not give good results
    # overall_data[np.where((max_cog_val == min_cog_val)), 0:1] = np.nan

    ind_row1 = np.where(((abs(max_cog_val - min_cog_val) > MINCOGCHANGE) ))[0]
    # ind_ano = ind_row1[np.where((abs(360. - max_cog_val[ind_row1] - min_cog_val[ind_row1]) > MINCOGCHANGE))[0]]
    # find false anomalies
    ind_ano1 = ind_row1[(max_cog_val[ind_row1] > 360 - MINCOGCHANGE) & (min_cog_val[ind_row1] < MINCOGCHANGE)]
    false_ano = np.array([])

    for j in ind_ano1:
        aa = np.where((overall_data[j, :] - MINCOGCHANGE <= 0))[0]
        bb = np.where((overall_data[j, :] - MINCOGCHANGE > 0))[0]
        max_cog_val_j = np.nanmax(overall_data[j, aa])  # finds max cog in 360+ data = angle rightside of North
        min_cog_val_j = abs(360 - np.nanmin(overall_data[j, bb]))  # finds min cog in 360- data and then angle leftside
        if max_cog_val_j + min_cog_val_j <= 30:
            false_ano = np.append(false_ano, np.where((ind_row1 == j))[0])

    # delete false anomalies
    ind_ano = np.delete(ind_row1, false_ano)

    ind_row_normal = np.delete(np.where(overall_data[:, 0:1] >= 0)[0], ind_ano)
    delete_normal = np.random.choice(np.arange(len(ind_row_normal)),overall_data.shape[0]-2*len(ind_ano), replace=False)
    overall_data[ind_row_normal[delete_normal], 0] = np.nan
    ind_row_normal = np.delete(ind_row_normal, delete_normal)
    # assign target values
    Y_data = np.zeros((overall_data.shape[0], CLASSES)) #
    Y_data[:, 0] = 1
    Y_data[ind_ano] = [0, 1]
    where_are_NaNs = np.isnan(overall_data[:, 0])
    Y_data[ where_are_NaNs, 0] = np.nan
    overall_data = overall_data[~where_are_NaNs]
    Y_data = Y_data[~where_are_NaNs]
    print('total number of normal samples is ', len(Y_data) - len(ind_ano))
    print('total number of anomaly samples is ', len(ind_ano))

    # overall_data[np.isnan(overall_data)] = Dummy_Nr
    return overall_data, Y_data


def load_test_data(timesteps, dim, track_to_check):
    path = '/home/sing_sd/Desktop/anomaly_detection/PythonCode/Resources/track_pickle/'

    filename = 'track{}'.format(track_to_check)

    try:
        data = pd.read_pickle(path + filename + '.pkl')
    except IOError:
        print("Error: File does not appear to exist for track ", track_to_check)
        return 0, 0
    # without interpolation

    #data = data[data['sog'] > MINSOG]
    data_per_track = data.shape[0]
    overall_data = np.full(shape=(data_per_track, timesteps*dim ), fill_value=np.nan)

    overall_data[:, 0:dim] = data.iloc[:, 2:6]

    # shift from top and put on remaining columns
    for clm_nr in range(1, timesteps):
        overall_data[0: data_per_track - 1, clm_nr * dim:(clm_nr + 1) * dim] = overall_data[1: data_per_track, (clm_nr - 1) * dim:clm_nr * dim]

    overall_data = overall_data[np.where(overall_data[:, -1] >= 0)[0]]

    # compute number of missing zeros after the first meassage in each row
    clm_ind = range(0, timesteps * dim, dim)  # cog_index is the column number in features array
    max_cog_val = np.nanmax(overall_data[:, clm_ind], axis=1)
    min_cog_val = np.nanmin(overall_data[:, clm_ind], axis=1)
    max_cog_val = max_cog_val.T
    min_cog_val = min_cog_val.T
    # find anomaly samples
    ind_row1 = np.where(((abs(max_cog_val - min_cog_val) > MINCOGCHANGE)))[0]
    # ind_ano = ind_row1[np.where((abs(360. - max_cog_val[ind_row1] - min_cog_val[ind_row1]) > MINCOGCHANGE))[0]]
    # find false anomalies
    ind_ano1 = ind_row1[(max_cog_val[ind_row1] > 360 - MINCOGCHANGE) & (min_cog_val[ind_row1] < MINCOGCHANGE)]
    false_ano = np.array([])

    for j in ind_ano1:
        aa = np.where((overall_data[j, :] - MINCOGCHANGE <= 0))[0]
        bb = np.where((overall_data[j, :] - MINCOGCHANGE > 0))[0]
        max_cog_val_j = np.nanmax(overall_data[j, aa])  # finds max cog in 360+ data = angle rightside of North
        min_cog_val_j = abs(360 - np.nanmin(overall_data[j, bb]))  # finds min cog in 360- data and then angle leftside
        if max_cog_val_j + min_cog_val_j <= 30:
            false_ano = np.append(false_ano, np.where((ind_row1 == j))[0])

    # delete false anomalies
    ind_ano = np.delete(ind_row1, false_ano)

    ind_row_normal = np.delete(np.where(overall_data[:, 0:1] >= 0)[0], ind_ano)
    delete_normal = np.random.choice(np.arange(len(ind_row_normal)), overall_data.shape[0] - 2 * len(ind_ano),
                                     replace=False)
    overall_data[ind_row_normal[delete_normal], 0] = np.nan
    # overall_data[ind_row_normal[0: overall_data.shape[0]-2*len(ind_ano)], 0] = np.nan

    # assign target values
    Y_data = np.zeros((overall_data.shape[0], CLASSES))
    Y_data[:, 0] = 1
    Y_data[ind_ano] = [0, 1]

    where_are_NaNs = np.isnan(overall_data[:, 0])
    Y_data[where_are_NaNs, 0] = np.nan
    overall_data = overall_data[~where_are_NaNs]
    Y_data = Y_data[~where_are_NaNs]

    print('total number of normal samples is ', len(Y_data) - len(ind_ano))
    print('total number of anomaly samples is ', len(ind_ano))
    return overall_data, Y_data

def load_test_data_time(timesteps, dim, features, track_to_check):
    path = '/home/sing_sd/Desktop/anomaly_detection/PythonCode/Resources/track_pickle/'

    filename = 'track{}'.format(track_to_check)

    try:
        data = pd.read_pickle(path + filename + '.pkl')

    except IOError:
        print("Error: File does not appear to exist for track ", track_to_check)
        return 0, 0
    data = data[data['sog'] > MINSOG]
    data = data.reset_index(drop=True)
    original_data = data
    start_time = datetime.strptime(data.iloc[0]['date'] + ' ' + data.iloc[0]['time'],
                                   '%m/%d/%Y %H:%M:%S')
    end_time = datetime.strptime(data.iloc[-1]['date'] + ' ' + data.iloc[-1]['time'],
                                 '%m/%d/%Y %H:%M:%S')

    data_per_track = int((end_time - start_time).total_seconds() // SAMPLING_TIME + 1)


    overall_data = np.full(shape=(data_per_track, timesteps * dim), fill_value=np.nan)
    temp_data = pd.DataFrame(index=range(data_per_track), columns=features, dtype=np.float)
    position_interpolated = pd.DataFrame(index=range(data_per_track), columns=["x","y"], dtype=np.float)
    for slot_index in range(0, data.shape[0]):  # //
        current_time = datetime.strptime(data.iloc[slot_index]['date'] + ' ' + data.iloc[slot_index]['time'],
                                         '%m/%d/%Y %H:%M:%S')
        index1 = int((current_time - start_time).total_seconds()) // SAMPLING_TIME
        temp_data.loc[index1, 0:dim] = data.loc[slot_index, features]
        position_interpolated.loc[index1, 0:2] = data.loc[slot_index, ["x","y"]]
    # interpolate
    temp_data = temp_data.fillna(method="ffill")

    position_interpolated = position_interpolated.interpolate(method='linear', limit_direction='forward', axis=0)
    # temp_data = temp_data.drop(temp_data.iloc[:,2] > 360)
    overall_data[:, 0:dim] = temp_data.iloc[:, 0:dim]

    # shift from top and put on remaining columns
    for clm_nr in range(1, timesteps):
        overall_data[0: data_per_track - 1, clm_nr * dim:(clm_nr + 1) * dim] = overall_data[1: data_per_track, (clm_nr - 1) * dim:clm_nr * dim]

    overall_data = overall_data[np.where(overall_data[:, -1] >= 0)[0]]
    # compute number of missing zeros after the first meassage in each row
    clm_ind = range(2, timesteps * dim, dim)
    max_cog_val = np.nanmax(overall_data[:, clm_ind], 1)
    min_cog_val = np.nanmin(overall_data[:, clm_ind], 1)
    max_cog_val = max_cog_val.T
    min_cog_val = min_cog_val.T
    ind_row1 = np.where(((abs(max_cog_val - min_cog_val) > MINCOGCHANGE)))[0]
    ind_ano = ind_row1[np.where((abs(360. - max_cog_val[ind_row1] - min_cog_val[ind_row1]) > MINCOGCHANGE))[0]]

    ind_row_normal = np.delete(np.where(overall_data[:, 0:1] >= 0)[0], ind_ano)

    # assign target values
    Y_data = np.zeros((overall_data.shape[0], CLASSES))
    Y_data[:, 0] = 1
    Y_data[ind_ano] = [0, 1]

    where_are_NaNs = np.isnan(overall_data[:, 0])
    overall_data = overall_data[~where_are_NaNs]
    Y_data = Y_data[~where_are_NaNs]
    print('total number of normal samples is ', len(Y_data) - len(ind_ano))
    print('total number of anomaly samples is ', len(ind_ano))

    overall_data[np.isnan(overall_data)] = Dummy_Nr
    return original_data, np.array(position_interpolated), overall_data, Y_data

def load_saved_data():
    with open("X_data.csv", 'rb') as f:
        X_data = pd.read_csv(f, sep=",", header=None)
    X_data = np.array(X_data)

    with open("Y_data.csv", 'rb') as f:
        Y_data = pd.read_csv(f, sep=",", header=None)
    Y_data = np.array(Y_data)

    return X_data, Y_data

# path = '/home/sing_sd/Desktop/anomaly_detection/PythonCode/Resources/track_pickle/'
# for track in range(1,229):
#     filename = 'track{}'.format(track)
#     try:
#         data = pd.read_pickle(path + filename + '.pkl')
#         data.to_csv(filename+".csv", index = False)
#     except IOError:
#         print("Error: File does not appear to exist for track ", track)