# NFLOPnet

Measurements and code for our paper "Near-field localization using machine learning: an empirical study" to be presented at VTC2021 Helsinki. 

Abstract:

"Estimation methods for passive near-field localization have been studied to an appreciable extent in signal processing research. Such localization methods find use in various applications, for instance in medical imaging. However, methods based on the standard near-field signal model can be inaccurate in real-world applications, due to deficiencies of the model itself and hardware imperfections. It is expected that deep neural network (DNN) based estimation methods trained on the nonideal sensor array signals could outperform the model-driven alternatives. In this work, a DNN based estimator is trained and validated on a set of real world measured data. The series of measurements was conducted with an inexpensive custom built multichannel software-defined radio (SDR) receiver, which makes the nonidealities more prominent. The results show that a DNN based localization estimator clearly outperforms the compared model-driven method."

A receiver based on https://github.com/mlaaks/coherent-rtlsdr was used in capturing the data. The 143 (grid 13 angles between -30:30 degrees, 11 distances between 0.5:5.5m) measured signals are in the folder 'signals'. Each .mat -file contains the measured complex samples and the ground truth distance & angle.

First you need to run the matlab script 

    computecovs.m 

then you can run the NFLOPnet.py python script which trains the network
on precomputed covariance matrices (computing the covariances in python directly was deemed too slow). 

Two matlab scripts process the results saved by NFLOPnet.py:          

    music_against_dnn_plot.m 
    music_against_dnn_stats.m
