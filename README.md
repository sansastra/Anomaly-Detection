# Anomaly-Detection
This repository contains python codes for detecting automatic identification system (AIS) on-off switching (OOS) anomalies, unusual turn detection, and deviation of trajectories of vessels.
AIS-OOS anomaly and unusual turn with more than 30 degrees are detected with deep neural networks.
Trajectory prediction is performed with Kalman Filter (KF) and a hybrid learning approach that uses a machine learning-based predictor (lstm or seq2seq) in the prediction steps. 
