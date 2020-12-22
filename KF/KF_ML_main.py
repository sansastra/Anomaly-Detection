import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from KF_functions import kf_update, kf_predict
from load_data import load_data_trajectory, convert2ENU, convert2Degree, predict_data

plt.rcParams['figure.figsize'] = (10, 8)

# intial parameters

dim_x = 6 # state dimension
dim_z = 3 # measurement dimension
INPUT_LEN = 10 # input length for prediction
# x = 0.37727 # truth value (typo in example at top of p. 13 calls this z)
track2predict = "Track167_sampled_1min.csv"

track_z = load_data_trajectory(track2predict)

sz = (track_z.shape[0], dim_x) # size of array
n_iter = track_z.shape[0]

zENU = convert2ENU(np.transpose(track_z[:,0]), np.transpose(track_z[:,1]))

plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(figsize=(8, 6))

# ax.plot(track_z[:, 0], track_z[:, 1], "k.")
# ax.plot(WGS84[0],WGS84[1],"r.")
# allocate space for arrays
xhat = []     # a posteri estimate of x
# P = np.zeros(sz)         # a posteri error estimate
# xhatminus = np.zeros(sz) # a priori estimate of x
# Pminus = np.zeros(sz)    # a priori error estimate
# K = np.zeros(sz)         # gain or blending factor
predicted = np.zeros((3, n_iter)) # E, N, U
estimated = np.zeros((3, n_iter)) # E, N, U
# intial guesses

# initialization
dt = 1
for i in range(INPUT_LEN):
    xhat.append(np.array([[zENU[0, i]], [zENU[1, i]], [zENU[2, i]], [1], [1], [1]]))

P = np.diag([1, 1, 1, 10*10, 10*10, 10*10]) # 1000*np.eye(dim_x)
A = np.array([[1, 0, 0, dt, 0, 0], [0, 1, 0, 0, dt, 0], [0, 0, 1, 0, 0, dt], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
Qbar = np.diag([0.5, 0.5, 0.1])  # 0.001*np.eye(dim_z) system or process error covariance
Fq = np.array([[dt*dt/2, 0, 0], [0, dt*dt/2, 0], [0, 0, dt*dt/2], [dt, 0, 0], [0, dt, 0], [0, 0, dt]])
Q = np.dot(Fq, np.dot(Qbar, np.transpose(Fq)))
B = np.eye(dim_x)
U = np.zeros((dim_x, 1))


# Measurement matrices
# Y = array([[X[0,0] + abs(randn(1)[0])], [X[1,0] +  abs(randn(1)[0])]])
H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
R = np.eye(dim_z) # eye(Y.shape[0]) # # estimate of measurement error covariance


for k in range(INPUT_LEN, n_iter):
    # time update or predict
    (xhatminus, Pminus) = kf_predict(xhat[k-1], P, A, Q, B, U)
    # get past INPUT_LEN data for prediction
    xhatminus[0], xhatminus[1], xhatminus[2], cog, sog = predict_data(xhat[k-INPUT_LEN:k], track_z[k-INPUT_LEN:k, :])
    predicted[0, k] = xhatminus[0]
    predicted[1, k] = xhatminus[1]
    predicted[2, k] = xhatminus[2]

    ax.plot(xhatminus[0], xhatminus[1], "k.")  # predicted trajectory
    # get measurement
    if not np.isnan(zENU[0, k]):
        z = np.array([[zENU[0, k]], [zENU[1, k]], [zENU[2, k]]])
        ax.plot(zENU[0, k], zENU[1, k], "rx")  # measured trajectory
    else:
        z = xhatminus[0:3]
        zENU[0, k] = xhatminus[0]
        zENU[1, k] = xhatminus[1]
        zENU[2, k] = xhatminus[2]
        track_z[k, 2] = cog
        track_z[k, 3] = sog

    # measurement update
    (xhat_new, P, K, IM, IS) = kf_update(xhatminus, Pminus, z, H, R)
    xhat.append(xhat_new)
    estimated[0, k] = xhat[k][0]
    estimated[1, k] = xhat[k][1]
    estimated[2, k] = xhat[k][2]

    ax.plot(xhat[k][0, 0], xhat[k][1, 0], "g.") # estimated
    plt.pause(0.001)

ax.plot(z[0], z[1], "rx", label="measured track")  # measured
ax.plot(xhat[0][0, 0], xhat[0][1, 0], "g.", label="Estimated")
ax.plot(xhatminus[0], xhatminus[1], "k.", label="predicted track")
ax.set(xlabel="x", ylabel="y")
plt.legend()
plt.pause(0.001)


fig, ax = plt.subplots()
plt.rcParams["legend.numpoints"] = 1
predict_xy = convert2Degree(predicted)
estimated_xy = convert2Degree(estimated)
ax.plot(track_z[:, 0], track_z[:, 1], 'k.', label="noisy measurements")
# ax.plot(estimated_xy[0, INPUT_LEN:], estimated_xy[1, INPUT_LEN:],'k.', label="a posteri estimate")
ax.plot(predict_xy[0, INPUT_LEN:], predict_xy[1, INPUT_LEN:], 'bx', label="a priori prediction")
ax.set(xlabel="Longitude (deg)", ylabel="Latitude (deg)")
ax.legend()
plt.pause(0.001)
plt.show()
# plt.figure()
# valid_iter = range(1,n_iter) # Pminus not valid at step 0
# plt.plot(valid_iter,Pminus[valid_iter],label='a priori error estimate')
# plt.title('Estimated $\it{\mathbf{a \ priori}}$ error vs. iteration step', fontweight='bold')
# plt.xlabel('Iteration')
# plt.ylabel('$(Voltage)^2$')
# plt.setp(plt.gca(),'ylim',[0,.01])
# plt.show()