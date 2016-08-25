import theano
import theano.tensor as T
import numpy as np
import os
import json
import socket
import cPickle as pickle
import argparse
import time
import datetime
import code

from layers.dynamic_filter_layer import DynamicFilterLayer

# lasagne
import lasagne
from lasagne.layers import EmbeddingLayer, DenseLayer, ReshapeLayer, ConcatLayer, Gate, LSTMLayer, DropoutLayer, SliceLayer, InputLayer, ElemwiseMergeLayer, NonlinearityLayer, FeaturePoolLayer, DimshuffleLayer, Upscale2DLayer, ElemwiseSumLayer, BiasLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayer
from lasagne.updates import rmsprop
#from lasagne.regularization import 
from lasagne.nonlinearities import softmax, identity, sigmoid, tanh, rectify, leaky_rectify
from lasagne.init import Uniform, Constant, Normal, HeUniform
from lasagne.utils import create_param

class Model(object):
    """ model initialization """
    def __init__(self, npx=64, batch_size=32, input_seqlen=1, target_seqlen=1, dynamic_filter_size=(9,9)):
        self.npx = npx
        self.batch_size = batch_size
        self.input_seqlen = input_seqlen
        self.target_seqlen = target_seqlen
        self.dynamic_filter_size = dynamic_filter_size

    def build_model(self, input_batch):
        filter_size = self.dynamic_filter_size[0]

        ## get inputs
        input = InputLayer(input_var=input_batch[:,[0],:,:], shape=(None, 1, self.npx, self.npx))
        theta = InputLayer(input_var=input_batch[:,[1],:,:], shape=(None, 1, self.npx, self.npx))
        # theta = ReshapeLayer(theta, shape=(self.batch_size, 1, 1, 1))

        output = ConvLayer(theta, num_filters=64, filter_size=(1,1), stride=(1,1), pad='same', nonlinearity=leaky_rectify)
        output = ConvLayer(output, num_filters=128, filter_size=(1, 1), stride=(1, 1), pad='same', nonlinearity=leaky_rectify)
        filters = ConvLayer(output, num_filters=filter_size ** 2, filter_size=(1, 1), stride=(1, 1), pad='same', nonlinearity=identity)

        image = SliceLayer(input, indices=slice(0, 1), axis=1)
        output = DynamicFilterLayer([image, filters], filter_size=(filter_size, filter_size, 1), pad=(filter_size // 2, filter_size // 2))

        return output, [output], filters