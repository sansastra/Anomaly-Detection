import numpy as np
from numpy.random import  randn
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

# First construct the object with the required dimensionality
f = KalmanFilter (dim_x=4, dim_z=2)

# Assign the initial value for the state (position and velocity). You can do this with a two dimensional array like so:
f.x = np.array([[0.0], [0.0], [0.1], [0.1]]) # [x, y, v_x, v_y]
# time step
dt = 1
# Define state transition matrix
f.F = np.array([[1, 0, dt , 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])

f.H = np.array([[1.,0.]])

# Define the covariance matrix. Here I take advantage of the fact that P already contains np.eye(dim_x), and just multiply by the uncertainty:
f.P *= 1000.

f.R = np.array([[5.]])
N_iter = 100
### Measurement
### create a real trajectory
X_Real = np.array(np.arange(0, N_iter))
Y_Real = X_Real/N_iter + 0.5


def get_sensor_reading(i):
    return np.array([[X_Real[i] + abs(0.1 * randn(1)[0])], [Y_Real[i] + abs(0.1 * randn(1)[0])]])


f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)

fig, ax = plt.subplots()

for i in np.arange(0, N_iter):
    ax.plot(X_Real[i], Y_Real[i], "k.")  # real trajectory
    z = get_sensor_reading(i)
    ax.plot(z[0, 0], z[1, 0], "gx")  # measured
    f.predict()
    f.update(z)
    ax.plot(f.x[0, 0], f.x[1, 0], "bo")  # estimated
    plt.pause(0.001)
    continue