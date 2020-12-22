import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from KF_functions import kf_update, kf_predict
from ENUtransform import WGS84toENU, ENUtoWGS84, WGS84toECEF

plt.rcParams['figure.figsize'] = (10, 8)

# intial parameters

dim_x = 6 # state dimension
dim_z = 3 # measurement dimension

# x = 0.37727 # truth value (typo in example at top of p. 13 calls this z)
path = '/home/sing_sd/Desktop/anomaly_detection/PythonCode/KF/'

filename1 = path + "Track167_sampled_1min.csv" # "Track167_interpolated_1min.csv"
# filename4 = path + "Track167_EKF.csv"
try:
   track_z = np.array(pd.read_csv(filename1))
   track_z[10:20, :] = np.nan
    # predicted_track_kf = np.array(pd.read_csv(filename4))
except IOError:
    print("Error: File does not appear to exist for track ")


lon =  np.transpose(track_z[:, 0])
lat = np.transpose(track_z[:, 1])
tREF = {"lon": 12.114733,
        "lat": 54.145409,
        "ECEF": np.array([[3660725], [785776], [514624]])
    }

zENU = WGS84toENU(lon, lat, tREF, h=0.)
# WGS84 = ENUtoWGS84(np.array(zENU), tREF)

sz = (track_z.shape[0], dim_x) # size of array
n_iter = track_z.shape[0]



# ax.plot(track_z[:, 0], track_z[:, 1], "k.")
# ax.plot(WGS84[0],WGS84[1],"r.")
# allocate space for arrays
# xhat = np.zeros(sz)      # a posteri estimate of x
# P = np.zeros(sz)         # a posteri error estimate
# xhatminus = np.zeros(sz) # a priori estimate of x
# Pminus = np.zeros(sz)    # a priori error estimate
# K = np.zeros(sz)         # gain or blending factor
predicted = np.zeros((3, n_iter)) # E, N, U
estimated = np.zeros((3, n_iter)) # E, N, U
# intial guesses

# initialization
dt = 1
xhat = np.array([[zENU[0, 0]], [zENU[1, 0]], [zENU[2, 0]], [0.5*track_z[0,3]*np.cos(track_z[0, 2])], [0.5*track_z[0,3]*np.sin(track_z[0, 2])], [0]])
P = np.diag([5**2, 5**2, 1, 10**2, 10**2, 0**2]) # 1000*np.eye(dim_x)
A = np.array([[1, 0, 0, dt, 0, 0], [0, 1, 0, 0, dt, 0], [0, 0, 1, 0, 0, dt], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
Qbar = np.diag([0.5, 0.5, 0.1])  # 0.001*np.eye(dim_z) system or process error covariance
Fq = np.array([[dt*dt*0.5, 0, 0], [0, dt*dt*0.5, 0], [0, 0, dt*dt*0.5], [dt, 0, 0], [0, dt, 0], [0, 0, dt]])
Q = np.dot(Fq, np.dot(Qbar, np.transpose(Fq)))
# Q = np.diag([5000**2, 5000**2, 1, 15**2, 15**2, 1]) # np.diag([500**2, 500**2, 1, 5.2**2, 5.2**2, 1]) is good when dt=1
B = np.eye(dim_x)
U = np.zeros((dim_x, 1))


# Measurement matrices
# Y = array([[X[0,0] + abs(randn(1)[0])], [X[1,0] +  abs(randn(1)[0])]])
H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
R = (3**2)*np.eye(dim_z) # eye(Y.shape[0]) # # estimate of measurement error covariance

plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(figsize=(8, 6))

for k in range(1, n_iter):
    # time update or predict
    (xhatminus, Pminus) = kf_predict(xhat, P, A, Q, B, U)
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

    # measurement update
    (xhat, P, K, IM, IS) = kf_update(xhatminus, Pminus, z, H, R)
    estimated[0, k] = xhat[0]
    estimated[1, k] = xhat[1]
    estimated[2, k] = xhat[2]

    ax.plot(xhat[0, 0], xhat[1, 0], "g.") # estimated
    plt.pause(0.001)

ax.plot(z[0], z[1], "rx", label="measured track")  # measured
ax.plot(xhat[0, 0], xhat[1, 0], "g.", label="Estimated")
ax.plot(xhatminus[0], xhatminus[1], "k.", label="predicted track")
ax.set(xlabel="x", ylabel="y")
plt.legend()
plt.pause(0.001)


fig, ax = plt.subplots()
plt.rcParams["legend.numpoints"] = 1
predict_xy = ENUtoWGS84(predicted, tREF)
estimated_xy = ENUtoWGS84(estimated, tREF)
ax.plot(track_z[:, 0], track_z[:, 1], 'k.', label="noisy measurements" )
# ax.plot(estimated_xy[0,1:], estimated_xy[1,1:],'k.', label="a posteri estimate")
ax.plot(predict_xy[0,1:], predict_xy[1,1:], 'bx', label="a priori prediction")
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