#!/usr/bin/env python3

import sys

CAFFE_WEIGHTS_FILE = "conv3d_deepnetA_sport1m_iter_1900000"
CONV_LAYERS = [1,4,7,9,12,14,17,19]
FC_LAYERS = [22,25,28]
OUTPUT = "keras_weights.pkl.xz"
FLIP_3D = True

import caffe_pb2 as pb
import numpy as np
import pkl_xz

print("Loading Caffe parameters")
caffe_params = pb.NetParameter()
caffe_params.ParseFromString(open(CAFFE_WEIGHTS_FILE,"rb").read())
keras_params=[]

# caffe implements correlation rather than convolution, must rotate weights 180
# degrees
def corr2conv(w):
    for i in range(3,5):
        w = np.flip(w,i)
    if FLIP_3D:
        w = np.flip(w,2)
    return w

def get_param_shape(l):
    return (l.num,l.channels,l.length,l.height,l.width)

def numpy_conv_weights(w):
    w = numpy_param(w)
    w = corr2conv(w)
    # Keras weight format is channels last
    w = np.transpose(w,(2,3,4,1,0))
    return w

def numpy_fc_weights(w):
    w = numpy_param(w)
    w = w.T
    return w

def first_fc_layer(w):
    # layer is 8192 -> 4096, with 8192 from channels first format 512,1,4,4
    # need channels last format 1,4,4,512
    assert w.shape == (8192,4096)
    return np.reshape(w,(512,4,4,4096)).transpose((1,2,0,3)).reshape((8192,4096))

def numpy_param(p):
    shape = get_param_shape(p)
    p = np.array(p.data).reshape(shape)
    p = np.squeeze(p)
    return p

print("Converting...")
for l in CONV_LAYERS + FC_LAYERS:
    w,b = caffe_params.layers[l].blobs
    w = numpy_fc_weights(w) if l in FC_LAYERS else numpy_conv_weights(w)
    if l == FC_LAYERS[0]:
        w = first_fc_layer(w)

    print(w.shape)
    b = numpy_param(b)
    print(b.shape)
    keras_params.append((w,b))
del caffe_params

print("Saving Keras parameters")
pkl_xz.save(keras_params,OUTPUT)
