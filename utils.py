# -*- coding: utf-8 -*-
"""
Created on Thu May 29 14:19:58 2019

@author: zhangzehua1

This file define some utils functions 

"""
import os
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from keras import backend as K
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.utils.vis_utils import plot_model as plot
from keras.layers import Input,Dense,Flatten,Reshape,Conv2D,Conv2DTranspose,BatchNormalization,LeakyReLU,Activation,UpSampling2D,MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD 

def auc(y_true, y_pred):
    try:
        auc = tf.metrics.auc(y_true, y_pred)[1]
        K.get_session().run(tf.local_variables_initializer())
        return auc
    except:
        y_true = tf.constant([1,0])
        y_pred = tf.constant([0,1])
        auc = tf.metrics.auc(y_true, y_pred)[1]
        K.get_session().run(tf.local_variables_initializer())
        return auc

def load_mnist(path = None):
    
    """Loads the MNIST dataset.
    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    
    assert path, "You need give a data path!"
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


