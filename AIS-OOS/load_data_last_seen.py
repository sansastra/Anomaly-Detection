import numpy as np
import pandas as pd
from math import sin, cos, sqrt, atan2, radians
from geopy import distance
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

power_outage = 70 # 74 # define max distance from AIS receiver for power outage
ROSTOCK = (12.114733, 54.145409)
MISS_NUM = -1
CLASSES = 3
EXTRA_FEATURES = 0 # one for distance, and another for time to last seen
SAMPLING_TIME = 2 # seconds
nr_of_vessels = 150


def _eq_(index_1, ano_rows, timesteps):
    for item in ano_rows:
        if index_1 <= item < index_1+timesteps:
            return True
def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges # each row represents start index of consecutive zeros, and end+1

def get_distance(coord_1):
    #print(geo.VincentyDistance(coord_1, ROSTOCK).km)
    return distance.distance(coord_1, ROSTOCK).km

    lat1 = radians(coordinate1[1])
    lon1 = radians(coordinate1[0])
    lat2 = radians(ROSTOCK[1])
    lon2 = radians(ROSTOCK[0])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return 6373.0 * c # earth radius


def get_po_boundary():

    boundary_x = np.arange(11.2, 12.66, 0.01)
    boundary_y = np.zeros(shape=(len(boundary_x), 1))
    i = 0
    for x1 in boundary_x:
        boundary_y[i] = ROSTOCK[1] + np.sqrt((power_outage / (1.85 * 60)) ** 2 - (ROSTOCK[0] - x1) ** 2)
        i += 1
    return boundary_x, boundary_y

def load_all_data(timesteps, dim, features, CLASSES):
    ############## generate data from pickle to train ANN ##
    np.random.seed(10)
    EXTRA_FEATURES = 0
    if CLASSES == 3:
        EXTRA_FEATURES = 1
    column_len = dim * timesteps + EXTRA_FEATURES
    ano_condition = 0.9 * timesteps*dim

    with open('all_tracks_processed.pkl', 'rb') as f:
        data = pickle.load(f)

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

    with open('data_len_track.pkl', 'rb') as f:
        data_all_tracks = pickle.load(f)
    vessel_nr = 18 #28
    overall_data = np.full(shape=(int(sum(data_all_tracks[:nr_of_vessels])+timesteps*len(data_all_tracks)+data.shape[0]), column_len), fill_value=-1.0)
    data1 = data.mmsi.unique()
    startIndex = 0

    for mmsi in data1[:nr_of_vessels]: #[vessel_nr:vessel_nr+1]: #

        decoded_mmsi = data[data['mmsi'] == mmsi]

        if decoded_mmsi.shape[0] > timesteps:
            decoded_mmsi = decoded_mmsi.reset_index(drop=True)


            # fraction_of_original_track = np.random.uniform(0.1, 0.3)
            # index_remove = np.delete(np.arange(1,decoded_mmsi.shape[0]), np.arange(1,decoded_mmsi.shape[0],step=10)) #np.delete(np.arange(1,decoded_mmsi.shape[0]),np.random.randint(2,decoded_mmsi.shape[0],int(fraction_of_original_track*decoded_mmsi.shape[0])))
            # decoded_mmsi.drop(index_remove, inplace=True)
            # decoded_mmsi = decoded_mmsi.reset_index(drop=True)

            start_time = datetime.strptime(decoded_mmsi.iloc[0]['date'] + ' ' + decoded_mmsi.iloc[0]['time'],
                                           '%m/%d/%Y %H:%M:%S')
            end_time = datetime.strptime(decoded_mmsi.iloc[-1]['date'] + ' ' + decoded_mmsi.iloc[-1]['time'],
                                         '%m/%d/%Y %H:%M:%S')
            #    old_s= old_s[len(old_s)-1:len(old_s)]+':'+ old_s[len(old_s)-1:len(old_s)]
            data_per_track = int((end_time - start_time).total_seconds() // SAMPLING_TIME + 1)
            #data_temp = np.full(shape=(data_per_track, column_len), fill_value=-1.0)
            # place ais info on first 4 columns

            for slot_index in range(0, decoded_mmsi.shape[0]):  # //
                current_time = datetime.strptime(
                    decoded_mmsi.iloc[slot_index]['date'] + ' ' + decoded_mmsi.iloc[slot_index]['time'], '%m/%d/%Y %H:%M:%S')
                index1 = int((current_time - start_time).total_seconds()) // SAMPLING_TIME
                overall_data[startIndex +index1, 0:dim] = decoded_mmsi.iloc[slot_index, 2:6]

            # shift from top and put on remaining columns
            for clm_nr in range(1, timesteps):
                overall_data[startIndex : startIndex + data_per_track  - clm_nr,
                clm_nr * dim:(clm_nr + 1) * dim] = overall_data[startIndex +1 : startIndex +data_per_track - clm_nr + 1, (clm_nr - 1) * dim:clm_nr * dim]
            # clear last few rows that does not have sufficient data to make ...
           # overall_data[data_per_track[vessel_nr]:data_per_track[vessel_nr] + timesteps,:] = MISS_NUM
           # overall_data[startIndex : startIndex + data_per_track, column_len - 1] = int(mmsi)
            overall_data[startIndex+data_per_track: startIndex + data_per_track + decoded_mmsi.shape[0], 0:dim] = decoded_mmsi.iloc[:, 2:6]
            startIndex += data_per_track + decoded_mmsi.shape[0]

    if (CLASSES == 2):
        overall_data = overall_data[np.where(overall_data[:, 0:1] >= 0)[0]]
        # compute number of missing zeros after the first meassage in each row
        index_clm = np.argmax(overall_data[:, dim:timesteps * dim] >= 0,
                              axis=1)  # np argmax returns 0 if a row has all data < 0
        index_zeros = np.where((index_clm == 0))[0]
        index_zero = index_zeros[np.where((overall_data[index_zeros, dim] == MISS_NUM))[0]]
        index_clm[index_zero] = (timesteps - 1) * dim
        ind_ano = np.where(index_clm >= ano_condition)[0]
        ind_normal = np.where(index_clm < ano_condition)[0]
        # column number (timesteps + 1) * dim  stores passed time since last received ais message

        # column number (timesteps + 1) * dim + 1 stores distance of vessels from Rostock

        # assign target values
        Y_data = np.zeros((overall_data.shape[0], CLASSES))
        Y_data[:, 0] = 1
        Y_data[ind_ano] = [0, 1]
        print('total number of anomaly samples is ', len(ind_ano))
        print('total number of normal samples is ', len(Y_data) - len(ind_ano))

        ############
        # distance_from_Rostock = np.floor(np.sqrt(
        #     (ROSTOCK[0] - overall_data[:, 0]) ** 2 + (ROSTOCK[1] - overall_data[:, 1]) ** 2) * 1.85 * 60)
        # overall_data[:, -1] = distance_from_Rostock
        # ind_po = np.where((distance_from_Rostock[ind_ano] >= power_outage))[0]
        # print('total number of original power outage samples is ', len(ind_po))


        #############
    else:
        index_zero = np.where(np.sum(overall_data == MISS_NUM, axis=1)== column_len)[0]

        ind_nonzero = np.where(overall_data[:,0:1] >= 0)[0]

        if (len(index_zero)> len(ind_nonzero)):
            new_data = overall_data[ind_nonzero[0:len(ind_nonzero)//2], 0:dim]
            new_data[0:len(ind_nonzero)//4,0] = np.random.uniform(11, 11.4, len(ind_nonzero)//4)
            new_data[0:len(ind_nonzero)//4, 1] = np.random.uniform(54, 55, len(ind_nonzero)//4)
            new_data[len(ind_nonzero) // 4:len(ind_nonzero) // 2, 0] = np.random.uniform(12.4, 13, len(ind_nonzero) // 2 -len(ind_nonzero) // 4)
            new_data[len(ind_nonzero) // 4:len(ind_nonzero) // 2, 1] = np.random.uniform(54.4, 55, len(ind_nonzero) // 2 -len(ind_nonzero) // 4)
            overall_data[index_zero[0:len(ind_nonzero)//2],0:dim] = new_data
        else:
            print('do something else')


        overall_data = overall_data[np.where(overall_data[:,0:1] >= 0)[0]]
        # compute number of missing zeros after the first meassage in each row
        index_clm = np.argmax(overall_data[:, dim:timesteps*dim] >= 0, axis = 1) # np argmax returns 0 if a row has all data < 0
        index_zeros = np.where((index_clm == 0))[0]
        index_zero = index_zeros[np.where((overall_data[index_zeros, dim] == MISS_NUM))[0]]
        index_clm[index_zero] = (timesteps-1)*dim
        ind_ano = np.where(index_clm >= ano_condition)[0]
        ind_normal = np.where(index_clm < ano_condition)[0]
        # column number (timesteps + 1) * dim  stores passed time since last received ais message

        # column number (timesteps + 1) * dim + 1 stores distance of vessels from Rostock

        # assign target values
        Y_data = np.zeros((overall_data.shape[0], CLASSES))
        Y_data[:, 0] = 1
        #distance_from_Rostock = np.floor(np.sqrt(
         #   (ROSTOCK[0] - overall_data[ind_ano, 0]) ** 2 + (ROSTOCK[1] - overall_data[ind_ano, 1]) ** 2) * 1.85 * 60)

        distance_from_Rostock = np.floor(np.sqrt(
            (ROSTOCK[0] - overall_data[:, 0]) ** 2 + (ROSTOCK[1] - overall_data[:, 1]) ** 2) * 1.85 * 60)
        overall_data[:,-1] = distance_from_Rostock
        ind_po = np.where((distance_from_Rostock[ind_ano] >= power_outage) )[0]
        Y_data[ind_ano[ind_po]]= [0, 1, 0]
        print('total number of power outage samples is ',len(ind_po))
        ind_ano1 = np.where((distance_from_Rostock[ind_ano] < power_outage) )[0]
        Y_data[ind_ano[ind_ano1]] = [0, 0, 1]
        print('total number of anomaly samples is ', len(ind_ano1))

        print('total number of normal samples is ', len(ind_normal))

    # plt.rcParams.update({'font.size': 16})
    # fig, ax = plt.subplots(figsize=(8, 7))
    # im = Image.open('world_11_54_12-4_55.PNG')  # in degrees and minutes
    # ax.imshow(im, extent=(11, 12.6666, 54.0, 55), aspect='auto')
    # ax.plot(overall_data[:,0], overall_data[:,1], 'w.',label='Trajectories of vessels')
    # boundary_x, boundary_y = get_po_boundary()
    # ax.plot(boundary_x, boundary_y, 'k-', markersize=8, label='AIS transmission reach')
    # ax.plot(ROSTOCK[0], ROSTOCK[1], 'ko', markersize=12, label='Rostock location')
    # plt.pause(0.01)
    # ax.plot(overall_data[ind_ano[ind_po], 0], overall_data[ind_ano[ind_po], 1], 'bo', markersize=10,label='Power outage')
    # ax.plot(overall_data[ind_ano[ind_ano1], 0], overall_data[ind_ano[ind_ano1], 1], 'rx', markersize=10,label='AIS anomaly')
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.title('AIS on-off switching anomaly detection')
    # ax.legend()
    # plt.show()
    # plt.pause(0.01)
    return overall_data, Y_data


