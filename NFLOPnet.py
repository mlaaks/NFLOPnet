#Near-field localization using neural networks, this is the python code for our paper
#Near-field localization using machine learning: an empirical study
#to be presented in VTC2021 Helsinki
#
#Mikko Laakso
#mikko.t.laakso@aalto.fi

import keras
from keras import models
from keras.models import Model
from keras import layers
from keras import optimizers
from keras.utils.vis_utils import plot_model
from keras.layers import Input, Reshape, Flatten, Dropout, Concatenate, Average, AveragePooling2D, MaxPooling2D, Lambda, Dense, BatchNormalization
from keras.models import model_from_json
from keras.layers import Layer

from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
from keras import backend as K
from keras import regularizers

import tensorflow as tf
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import cmath

import os
from scipy.io import loadmat
from scipy.io import savemat


#function to fetch data from matlab generated covariance .mat files.
def flattendata(r_all,y_all,RM_all,R,RM,thetak,rk,N):
	M = R.shape[1]
	#R = np.hstack((R,rk)) #include true distance
	try:
		r_all = np.vstack((r_all,R))
		y_all = np.vstack((y_all,np.hstack((thetak,rk))))
		RM_all = np.concatenate((RM_all,RM),axis=0)
	except:
		r_all = R
		y_all = np.hstack((thetak,rk))
		RM_all= RM

	return (r_all,RM_all,y_all)


#Load the precomputed covariances and true loc vectors
X_all = np.array([])
y_all = np.array([])
RM_all = np.array([])
for fname in os.listdir('covariances/'):
	print('opening: ' + fname + '...')
	dataset = loadmat('covariances/'+fname)
	r_v   = dataset['r_v']
	rk 	  = dataset['r_k']
	thetak= dataset['theta_k']
	N = dataset['snapshots']
	RM = dataset['RM']
	RM = np.swapaxes(RM,0,2)
	RM = np.swapaxes(RM,1,2)
	(X_all,RM_all,y_all) = flattendata(X_all,y_all,RM_all,r_v,RM,thetak,rk,N)

#to better see if we're actually running on the GPU:
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print(X_all.shape)
X_tr, X_te, y_tr, y_te = train_test_split(X_all,y_all,test_size=0.1, random_state=42)
RM_tr,RM_te,y2_tr,y2_te = train_test_split(RM_all,y_all,test_size=0.1,random_state=42)


#Define the NFLOPnet model with functional api, although the basic api would have sufficed here:

#Layers:
input_l = Input(shape=(X_tr.shape[1],))
bn = BatchNormalization()(input_l)
encoder_l  = Dense(X_tr.shape[1],activation='relu')(bn)
hidden_l  = Dense(512,activation='relu')(encoder_l) #512
hidden_l2  = Dense(64,activation='elu')(hidden_l) #128 , was 162 initially,64 deemed best
final_l = Dense(2,activation='elu')(hidden_l)

#Compile the model
model = Model(inputs = input_l, outputs = final_l)
model.summary()
plot_model(model,to_file='NFLOPnet.png',show_shapes=True,show_layer_names=True)
model.compile(metrics=['accuracy'],loss='mean_squared_error',optimizer='adam')


#train the model and run predict on the test set
model.fit(X_tr,y_tr,epochs=1024,batch_size=1024,verbose=1)
y_pred = model.predict(X_te)


# SAVE THE TRAINED MODEL:
model_json = model.to_json()
with open("NFLOPnet.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("NFLOPnet.h5")

#save in matlab format for plotting (neater images than python libs)
savemat('errsd.mat',{'y_pred':y_pred,'y_te':y_te})
savemat('covsn_predicts.mat',{'y_pred':y_pred,'y_te':y_te,'R_te':RM_te}) #save the full covariance so we can compare with MUSIC.


#compute some error stats, print info
N    = y_te.shape[0]
MSEd = np.sum(np.power((y_te[:,0]-y_pred[:,0]),2))/N
MSEa = np.sum(np.power((y_te[:,1]-y_pred[:,1]),2))/N
MAEd = np.sum(np.abs(y_te[:,0]-y_pred[:,0]))/N
MAEa = np.sum(np.abs(y_te[:,1]-y_pred[:,1]))/N

print('Training set N ={trn}, Test set N = {tsn} '.format(trn=X_tr.shape[0], tsn=X_te.shape[0]))
print('MSE angle = {ea} [deg], MSE distance = {ed} [cm]'.format(ea=180/np.pi*MSEa,ed=100*MSEd))
print('MAE angle = {ea} [deg], MAE distance = {ed} [cm]'.format(ea=180/np.pi*MAEa,ed=100*MAEd))
print('MSE angle = {ea} [deg], MSE distance = {ed} [lambda]'.format(ea=180/np.pi*MSEa,ed=MSEd/(300/1240)))
print('MAE angle = {ea} [deg], MAE distance = {ed} [lambda]'.format(ea=180/np.pi*MAEa,ed=MAEd/(300/1240)))