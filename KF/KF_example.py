# Kalman filter example demo in Python

# A Python implementation of the example given in pages 11-15 of "An
# Introduction to the Kalman Filter" by Greg Welch and Gary Bishop,
# University of North Carolina at Chapel Hill, Department of Computer
# Science, TR 95-041,
# https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf

# by Andrew D. Straw

import numpy as np
import matplotlib.pyplot as plt
from KF_functions import kf_update, kf_predict

plt.rcParams['figure.figsize'] = (10, 8)

# intial parameters
n_iter = 100
dim_x = 4 # state dimension
dim_z = 2 # measurement dimension
sz = (n_iter, dim_x) # size of array
# x = 0.37727 # truth value (typo in example at top of p. 13 calls this z)
x = [(np.arange(0, n_iter)), 2 * (np.arange(0, n_iter)/n_iter) + 5]

z_x = np.random.normal(x[0], 1, size=sz[0]) # observations (normal about x, sigma=0.1)
z_y = np.random.normal(x[1], 1, size=sz[0]) # observations (normal about x, sigma=0.1)


# allocate space for arrays
# xhat = np.zeros(sz)      # a posteri estimate of x
# P = np.zeros(sz)         # a posteri error estimate
# xhatminus = np.zeros(sz) # a priori estimate of x
# Pminus = np.zeros(sz)    # a priori error estimate
# K = np.zeros(sz)         # gain or blending factor

# intial guesses


# Initialization of state matrices
dt = 1
xhat = np.array([[0.0], [4.5], [1], [0.02]])
P = 100*np.eye(dim_x)
A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
Q = 0.0001*np.eye(dim_x) # system error
B = np.eye(dim_x)
U = np.zeros((dim_x, 1))

# Measurement matrices
# Y = array([[X[0,0] + abs(randn(1)[0])], [X[1,0] +  abs(randn(1)[0])]])
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
R = 1000*np.eye(dim_z) # eye(Y.shape[0]) # # uncertainty in estimate of measurement variance, change to see effect

fig, ax = plt.subplots()
for k in range(1, n_iter):
    # time update or predict
    (xhatminus, Pminus) = kf_predict(xhat, P, A, Q, B, U)
    ax.plot(x[0][k], x[1][k], "k.")  # real trajectory
    # get measurement
    z = np.array([[z_x[k]], [z_y[k]]])
    ax.plot(z[0], z[1], "rx")  # measured trajectory
    # measurement update
    (xhat, P, K, IM, IS) = kf_update(xhatminus, Pminus, z, H, R)
    ax.plot(xhat[0, 0], xhat[1, 0], "go") # estimated



ax.plot(x[0][0], x[1][0], "k.", label="real track")
ax.plot(z[0], z[1], "rx", label="measured track")  # measured
ax.plot(xhat[0, 0], xhat[1, 0], "go", label = "Estimated")
ax.set(xlabel = "x", ylabel = "y")
plt.legend()
plt.pause(0.001)
plt.show()

plt.figure()
plt.plot(z,'k+',label='noisy measurements')
plt.plot(xhat,'b-',label='a posteri estimate')
plt.axhline(x,color='g',label='truth value')
plt.legend()
plt.title('Estimate vs. iteration step', fontweight='bold')
plt.xlabel('Iteration')
plt.ylabel('Voltage')

plt.figure()
valid_iter = range(1,n_iter) # Pminus not valid at step 0
plt.plot(valid_iter,Pminus[valid_iter],label='a priori error estimate')
plt.title('Estimated $\it{\mathbf{a \ priori}}$ error vs. iteration step', fontweight='bold')
plt.xlabel('Iteration')
plt.ylabel('$(Voltage)^2$')
plt.setp(plt.gca(),'ylim',[0,.01])
plt.show()