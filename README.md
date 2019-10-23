# DCGAN_keras_tf_backend

[TOC]

This repo contain a DCGAN implement with keras which is tensorflow backend in **dcgan.py**. 

Besides, a mnist demo is provided in **train.py**.



## HOW TO USE

It's easy to declare a DCGAN model class use

```python
from dcgan import *
from keras.optimizers import SGD 

model = DCGAN(g_depths=[1024,512,256,1], d_depths=[1,256,512], g_size=7, d_size=28)

#compile
optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
model.g.compile(loss='binary_crossentropy', optimizer='SGD')
model.gan.compile(loss='binary_crossentropy', optimizer=optim) 
model.d.trainable = True
model.d.compile(loss='binary_crossentropy', optimizer=optim, metrics=['acc',auc])
```

* g_depths: Generator feature depth of each layer
* d_depths: Discriminator feature depth of each layer
* g_size: base size, the output image's size will be size*2**(len(depths)-1)
* d_size: input size



## DEMO

***Perhaps you need modify GPU config code in train.py Line 139***

**TRAIN**

```shell
python train.py
```

you will get Generator, Discriminator and DCGAN summary.

The metrics is G:[loss], D:[loss, acc, auc]

* genrated image will be saved at ./genrate_image
* Generator, Discriminator and DCGAN model will be saved at ./model

