from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import math
from dcgan import *
import pdb
from keras.optimizers import SGD 
import utils


#concat output of predict as a big image(size=[sqrt(batch)*image_wight,sqrt(batch)*image_hight])
def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image


def train(BATCH_SIZE):
    #(X_train, y_train), (X_test, y_test) = mnist.load_data()
    (X_train, y_train), (X_test, y_test) = utils.load_mnist(path = './mnist_data/mnist.npz')
    #for easy to caculate? same as /255
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train[:, :, :, None]
    X_test = X_test[:, :, :, None]
    # X_train = X_train.reshape((X_train.shape, 1) + X_train.shape[1:])
    
    model = DCGAN(g_depths=[1024,512,256,1], d_depths=[1,256,512], g_size=7, d_size=28)

    #compile
    optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    model.g.compile(loss='binary_crossentropy', optimizer='SGD')
    model.gan.compile(loss='binary_crossentropy', optimizer=optim) 
    model.d.trainable = True
    model.d.compile(loss='binary_crossentropy', optimizer=optim, metrics=['acc',auc])


    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = model.g.predict(noise, verbose=0)
            if index % 20 == 0:
                image = combine_images(generated_images)
                #recover gray value
                #pdb.set_trace()
                #print (image)
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    "./genrate_image/"+str(epoch)+"_"+str(index)+".png")
                #print (image)
            #pdb.set_trace()
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss, acc, auc_v = model.d.train_on_batch(X, y)
            
            #pdb.set_trace()
            print("epoch %d batch %d: [d_loss = %f, acc = %f, auc = %f]" \
            % (epoch, index, d_loss, acc, auc_v))
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            model.d.trainable = False
            #model.d.compile(optimizer=optim, loss='binary_crossentropy', metrics=['acc',auc])

            #pdb.set_trace()
            g_loss = model.gan.train_on_batch(noise, [1] * BATCH_SIZE)
            
            model.d.trainable = True
            #model.d.compile(optimizer=optim, loss='binary_crossentropy', metrics=['acc',auc])
            #pdb.set_trace()

            print("epoch %d batch %d: [g_loss = %f]" % (epoch, index, g_loss))
            
        
        model.g.save_weights('./model/generator.model', True)
        model.d.save_weights('./model/discriminator.model', True)
        model.gan.save_weights('./model/gan.model', True)

def generate(BATCH_SIZE, nice=False):
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')
    if nice:
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"]='7'
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
