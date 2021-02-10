This folder contains the dataset and code used for the paper Near-field localization using machine learning: an empirical study to be presented at VTC2021.

The 143 (grid 13 angles between -30:30 degrees, 11 distances between 0.5:5.5m) measured signals are in the folder 'signals'. Each .mat -file contains the measured complex samples and the ground truth distance & angle.

First you need to run the matlab script 

    computecovs.m 

then you can run the NFLOPnet.py python script which trains the network
on precomputed covariance matrices (computing the covariances in python directly was deemed too slow). 

Two matlab scripts process the results saved by NFLOPnet.py:          

    music_against_dnn_plot.m 
    music_against_dnn_stats.m
