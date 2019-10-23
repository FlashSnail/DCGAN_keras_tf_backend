# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:21:58 2019

@author: zhangzehua1

This file define a DCGAN model with keras 

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

class Generator:
    def __init__(self, depths=[1024,512,256,128,1], size=4, noise_dim=100):
        '''
        #depths[-1] : final depth you want to output
        
        #size : base size, the output will be size*2**(len(depths)-1)
        '''
        
        self.depths = depths
        self.size = size
        self.noise_dim = noise_dim
        
    def get_model(self, inputs = None, training=False):
        #input
        inputs = Input(shape=(self.noise_dim,))

        #reshape
        model = Dense(self.depths[0])(inputs)
        model = Activation('tanh')(model)
        model = Dense(self.size * self.size * self.depths[1])(model)
        model = BatchNormalization()(model)
        model = Activation('tanh')(model)
        model = Reshape((self.size, self.size, self.depths[1]))(model)
        
        #deconv
        for depth in self.depths[2:]:
            model = self.deconv(model, depth, (5,5), (2,2))
            #model = self.upsample(model, depth)

        #output
        outputs = model
        
        model = keras.models.Model(inputs,outputs)
        adam = keras.optimizers.Adam()
        #model.compile(optimizer='SGD', loss='binary_crossentropy')
        return model
#        return outputs
        
    def deconv(self, layer, filters, kernel_size, strides, padding='same'):
        x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(layer)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)
        return x

    def upsample(self, layer, filters):
        x = UpSampling2D(size=(2, 2))(layer)
        x = Conv2D(filters=filters, kernel_size = (5,5), padding='same')(x)
        x = Activation('tanh')(x)
        return x



class Discriminator:
    def __init__(self, depths=[1,64,128,256,512], size=64):
        '''
        #depths[-1] : final depth you want to output
        
        #size : input size
        '''
        
        self.depths = depths
        self.size = size
    
    def get_model(self, inputs = None, training=False):
        #input
        inputs = Input(shape=(self.size, self.size, self.depths[0]))
        #conv
        model = self.conv(inputs, self.depths[1], (5,5))
        #for depth in self.depths[2:3]:
        #    model = self.conv(model, depth, (5,5))
        #model = self.conv(model, self.depths[2], (5,5), padding='valid')
        model = self.conv(model, self.depths[2], (5,5))
        #output
        model = Flatten()(model)
        model = Dense(1024)(model)
        model = Activation('tanh')(model)
        model = Dense(1)(model)
        outputs = Activation('sigmoid')(model)
        
        model = keras.models.Model(inputs,outputs)
        adam = keras.optimizers.Adam()
        d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        #model.compile(optimizer=d_optim, loss='binary_crossentropy',metrics=['acc',auc])
        return model
            
    def conv(self, layer, filters, kernel_size, strides=1, padding='same'):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(layer)
        #x = BatchNormalization()(x)
        #x = LeakyReLU(alpha=0.2)(x)
        x = Activation('tanh')(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        return x

class DCGAN:
    def __init__(self,
                 batch_size=128, size=64, noise_dim=100,
                 g_depths=[1024,512,256,128,1], g_size=4,
                 d_depths=[1,64,128,256,512], d_size=64):
        self.batch_size = batch_size
        self.g_size = g_size
        self.d_size = d_size
        self.noise_dim = noise_dim
        self.g = Generator(depths=g_depths, size=self.g_size, noise_dim=self.noise_dim).get_model()
        print('\n======================Generator summary======================')
        self.g.summary()
        self.d = Discriminator(depths=d_depths, size=self.d_size).get_model()
        print('\n======================Discriminator summary======================')
        self.d.summary()
        self.gan = self.get_model()
        print('\n======================GAN summary======================')
        self.gan.summary()
        
    def get_model(self):
        model = Sequential()
        model.add(self.g)
        self.d.trainable = False
        model.add(self.d)
        adam = keras.optimizers.Adam()
        d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        #self.d.compile(optimizer=d_optim, loss='binary_crossentropy', metrics=['acc',auc])
        #model.compile(optimizer=g_optim, loss='binary_crossentropy')
        return model
   

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


