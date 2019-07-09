# ====================================================================================================================================================== 
#!/usr/bin/env python
# coding: utf-8
# Angle of Arrival Estimation of Uniform Linear and Cicular Array
# Jianyuan Jet Yu, jianyuan@vt.edu

# =============== 

import yagmail
from myFunction import myCompress
import time
import os
import math




# ======================== global setting ======================== 
dim          = 1;
isSend       = 1     # send Email
nb_epoch     = 20    # number of learning epochs
batch_sizes = 300
os.environ["KMP_DUPLICATE_LIB_OK"] = "1"   # sometime make "1" for Mac 

TEXT    = "test" 
num_bins = 50

tic = time.time()


import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
def get_session(gpu_fraction=1):
    
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())
import numpy as np
import hdf5storage
from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Lambda, Reshape, Conv1D, Conv2D, AveragePooling2D,Flatten, Dropout, SimpleRNN, LSTM, concatenate
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from IPython.core.display import Image, display
import matplotlib.pyplot as plt
from numpy import ma
import scipy.io as sio
from IPython.display import Image
from matplotlib import cm as CM
from nbconvert import HTMLExporter
import keras
keras.__version__

# Visualize training history
from keras import callbacks
tb = callbacks.TensorBoard(log_dir='./logs', histogram_freq=10, batch_size=32,
                           write_graph=True, write_grads=True, write_images=False,
                           embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
# Early stopping  
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')



#!nvidia-smi

# Distance Functions
#def dist(y_true, y_pred):    
#    return tf.reduce_mean((tf.sqrt(tf.square(tf.abs(y_pred[:,0]-y_true[:,0]))+tf.square(tf.abs(y_pred[:,1]-y_true[:,1]))+tf.square(tf.abs(y_pred[:,2]-y_true[:,2])))))

def dist(y_true, y_pred):    
    return tf.reduce_mean( tf.sqrt(tf.square(tf.abs(y_pred-y_true))))


#def true_dist(y_true, y_pred):    
#    return (np.sqrt(np.square(np.abs(y_pred[:,0]-y_true[:,0]))+np.square(np.abs(y_pred[:,1]-y_true[:,1]))+np.square(np.abs(y_pred[:,2]-y_true[:,2]))))
def true_dist(y_true, y_pred):
    return (np.sqrt(np.square(np.abs(y_pred-y_true))))


#  =============== 

# Example for Measurement Quality

t = hdf5storage.loadmat('./data/testXtrain.mat')
Xtrain = np.transpose(t['Xtrain'])
t = hdf5storage.loadmat('./data/testYtrain.mat')
Ytrain = np.transpose(t['Ytrain'])
t = hdf5storage.loadmat('./data/testXtest.mat')
Xtest = np.transpose(t['Xtest'])
t = hdf5storage.loadmat('./data/testYtest.mat')
Ytest = np.transpose(t['Ytest'])


Xval = Xtrain[0:300,:];
Yval = Ytrain[0:300,:];
Xtrain = Xtrain[301:,:];
Ytrain = Ytrain[301:,:];

# extend dim
Xtrain = Xtrain[:,:,np.newaxis]
Xtest = Xtest[:,:,np.newaxis]
Xval = Xval[:,:,np.newaxis]



nn_input  = Input((78,1))
            
nn_output = Flatten()(nn_input)
# nn_output = LSTM(units=136, dropout_U = 0.2, dropout_W = 0.2)(nn_input)
nn_output = Dense(256,activation='relu', kernel_regularizer=regularizers.l2(0.001))(nn_output)
nn_output = Dropout(0.2)(nn_output)
nn_output = BatchNormalization()(nn_output)
nn_output = Dense(128,activation='relu', kernel_regularizer=regularizers.l2(0.001))(nn_output)
nn_output = Dropout(0.2)(nn_output)
nn_output = BatchNormalization()(nn_output)
nn_output = Dense(64,activation='relu', kernel_regularizer=regularizers.l2(0.001))(nn_output)
nn_output = Dropout(0.2)(nn_output)
nn_output = BatchNormalization()(nn_output)
nn_output = Dense(1,activation='linear')(nn_output)
#  directly output 3 para, non classify
nn = Model(inputs=nn_input,outputs=nn_output)

nn.compile(optimizer='Adam', loss='mse',metrics=[dist])
nn.summary()
 

train_hist = nn.fit(x=Xtrain,y=Ytrain,\
                    batch_size= 300 ,epochs= 30,
                    validation_data=(Xval, Yval), \
                    shuffle=True,\
                    callbacks=[tb, early_stop])
        
# Evaluate Performance
Ypredtrain = nn.predict( Xtrain)
Ypred = nn.predict( Xtest)

Ypredtrain = Ypredtrain/math.pi*180
Ypred = Ypred/math.pi*180
Ytest = Ytest/math.pi*180
Ytrain = Ytrain/math.pi*180

errors_train = true_dist(Ytrain,Ypredtrain)
errors_test  = true_dist(Ytest,Ypred)
   
Mean_Error_Train =  np.mean(np.abs(errors_train))
Mean_Error_Test= np.mean(np.abs(errors_test))
print("Mean error on Train area:", Mean_Error_Train)
print("Mean error on Test  area:",Mean_Error_Test)


toc =  time.time()
timeCost = toc - tic
print( "--- Totally %s seconds ---" %(timeCost))



# Histogramm of errors on test Area
plt.figure(2)
errors = true_dist(Ypred , Ytest)
plt.hist(errors,bins=64)
plt.ylabel('Number of occurence')
plt.xlabel('Estimate error (deg)')
plt.grid(True)  
plt.title('histogram of estimation error')
plt.savefig('err_occ.png')

num_bins = 50
plt.figure(3)
counts, bin_edges = np.histogram(errors, bins=num_bins)
cdf = np.cumsum(counts)/np.sum(counts)
plt.plot(bin_edges[1:], cdf)
plt.xlabel('est error (deg)')
plt.ylabel('F(X<x)')
plt.grid(True)  
plt.title('Cdfplot of distance error')
plt.savefig('err_cdf.png')


#  =============== 
# Error Vector over Area in XY
#error_vectors = np.real(Ypred - Ytest)
#plt.figure(4)
#plt.quiver(np.real(Ytest[:,0]),np.real(Ytest[:,1]),error_vectors[:,0],error_vectors[:,1],errors)
#plt.xlabel("x in m")
#plt.ylabel("y in m")
#plt.grid(True) 
#plt.title('quiver of distance error') 
#plt.savefig('position.png')


plt.figure(5)
plt.plot(train_hist.history['dist'])
plt.plot(train_hist.history['val_dist'])
plt.title('distance')
plt.ylabel('distance')
plt.xlabel('epoch')
plt.grid(True)  
plt.legend(['train', 'validate'])
plt.savefig('hist_dist.png')

plt.figure(6)
plt.plot(train_hist.history['loss'])
plt.plot(train_hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.grid(True)  
plt.legend(['train', 'validate'])
plt.savefig('hist_loss.png')

dim = 1

plt.figure(7)
plt.scatter( Ytrain, Ypredtrain,facecolors='none',edgecolors='b')
plt.title('dim%d train - est vs ground'%dim)
plt.ylabel('est')
plt.xlabel('ground')
plt.grid(True)  
plt.savefig('dim%d_train.png'%dim)

plt.figure(8)
plt.scatter( Ytest, Ypred,facecolors='none',edgecolors='b')
plt.title('dim%d test - est vs ground'%dim)
plt.ylabel('est')
plt.xlabel('ground')
plt.grid(True)  
plt.savefig('dim%d_test.png'%dim)


#  =============== save result ===========

with open("outPut.txt", "w") as text_file:
    text_file.write( "--- Totally %s seconds --- \n" %(timeCost))
    text_file.write("Mean error on Train area:  %s \n" %(Mean_Error_Train))
    text_file.write("Mean error on Test  area: %s \n" %(Mean_Error_Test))

#  =============== send email ===========
FROM = 'yujianyuanhaha@gmail.com'
TO = 'yujianyuanhaha@gmail.com'

SUBJECT = 'CTW train err %.4f, test err %.4f'%(Mean_Error_Train, Mean_Error_Test)

ATTACHMENTS = ['dim%d_test.png'%dim,'dim%d_train.png'%dim,
               'outPut.txt','cdf.png','err_occ.png','err_cdf.png',\
               'hist_dist.png','hist_loss.png','main.py'
               ]

yag = yagmail.SMTP(FROM, 'yu3843526')
if isSend:
    yag.send(TO, SUBJECT, TEXT,ATTACHMENTS)
# ====================================================================================================================================================== 
