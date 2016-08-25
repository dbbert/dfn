# -*- coding: utf-8 -*-


import numpy as np
from collections import OrderedDict
import sys
import os

import theano
import theano.tensor as T

import lasagne
from lasagne import layers
from lasagne import init
from lasagne import nonlinearities
from lasagne.layers.base import Layer, MergeLayer

from lasagne.layers.conv import conv_output_length
from lasagne.layers.pool import pool_output_length
from lasagne.utils import as_tuple

from theano.sandbox.cuda import dnn # xu

__all__ = [
    "DynamicFilterLayer"
]

# class Deconv2DLayer(lasagne.layers.Layer):
#     """ deconv layer from Jan Schlueter """
#     def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0,W=init.GlorotUniform(),
#                  b=init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
#         super(Deconv2DLayer, self).__init__(incoming, **kwargs)
#         self.num_filters = num_filters
#         self.filter_size = lasagne.utils.as_tuple(filter_size, 2, int)
#         self.stride = lasagne.utils.as_tuple(stride, 2, int)
#         self.pad = lasagne.utils.as_tuple(pad, 2, int)
#         self.W = self.add_param(W,(self.input_shape[1], num_filters) + self.filter_size, name='W')
#         if b is None:
#             self.b = None
#         else:
#             if self.untie_biases:
#                 biases_shape = (num_filters, self.output_shape[2],
#                                 self.output_shape[3])
#             else:
#                 biases_shape = (num_filters,)
#             self.b = self.add_param(b, biases_shape, name="b",
#                                     regularizable=False)
#         if nonlinearity is None:
#             nonlinearity = lasagne.nonlinearities.identity
#         self.nonlinearity = nonlinearity
#
#     def get_output_shape_for(self, input_shape):
#         shape = tuple(i*s - 2*p + f - 1
#                 for i, s, p, f in zip(input_shape[2:],
#                                       self.stride,
#                                       self.pad,
#                                       self.filter_size))
#         return (input_shape[0], self.num_filters) + shape
#
#     def get_output_for(self, input, **kwargs):
#         op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
#             imshp=self.output_shape,
#             kshp=(self.input_shape[1], self.num_filters) + self.filter_size,
#             subsample=self.stride, border_mode=self.pad)
#         conved = op(self.W, input, self.output_shape[2:])
#         if self.b is not None:
#             conved += self.b.dimshuffle('x', 0, 'x', 'x')
#         return self.nonlinearity(conved)
# #
# class DynConvLayer(MergeLayer):
#     '''
#     input : X (batch_num, 1, w,h), W (batch,nf,wk,hk)
#     output :
#     '''
#     def __init__(self, incomings, stride=1, pad=0,
#         nonlinearity=nonlinearities.rectify, flip_filters=False, **kwargs):
#         super(DynConvLayer, self).__init__(incomings, **kwargs)
#         if nonlinearity is None:
#             self.nonlinearity = nonlinearities.identity
#         else:
#             self.nonlinearity = nonlinearity
#         n = len(self.input_shapes[0]) - 2 # or n=2
#         self.n = n
#         self.nc = self.input_shapes[0][1] # nc=1
#         self.nf = self.input_shapes[1][1]
#         self.filter_size = as_tuple(self.input_shapes[1][2], n, int)
#         self.flip_filters = flip_filters
#         self.stride = as_tuple(stride, n, int)
#
#         if pad == 'same':
#             if any(s % 2 == 0 for s in self.filter_size):
#                 raise NotImplementedError(
#                     '`same` padding requires odd filter size.')
#         if pad == 'valid':
#             self.pad = as_tuple(0, n)
#         elif pad in ('full', 'same'):
#             self.pad = pad
#         else:
#             self.pad = as_tuple(pad, n, int)
#
#     def get_output_shape_for(self, input_shapes):
#         # refer to lasagne.dnn layer
#         pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * self.n
#         batchsize = input_shapes[0][0]
#
#         return ((batchsize, 1, self.nf) +
#                 tuple(conv_output_length(input, filter, stride, p)
#                       for input, filter, stride, p
#                       in zip(input_shapes[0][2:], self.filter_size,
#                              self.stride, pad)))
#
#     def get_output_for(self, inputs, **kwargs):
#         # define a function and apply the same operation to each sample
#         # using scan
#         conv_mode = 'conv' if self.flip_filters else 'cross'
#         border_mode = self.pad
#         if border_mode == 'same':
#             border_mode = tuple(s // 2 for s in self.filter_size)
#
#         def onesample_conv(x,w):
#             x_ = T.reshape(x, (1, self.nc, x.shape[1], x.shape[2]))
#             w_ = T.reshape(w, (self.nf, self.nc, w.shape[1], w.shape[2]))
#             conved = dnn.dnn_conv(img=x_,
#                               kerns=w_,
#                               subsample=self.stride,
#                               border_mode=border_mode,
#                               conv_mode=conv_mode
#                               )
#             return conved
#         output,_ = theano.scan(onesample_conv, sequences=[inputs[0],inputs[1]])
#         return self.nonlinearity(output)
# #
# class LocalExpandLayer(lasagne.layers.Layer):
#     '''
#     used later in order to implement local connected layer
#     input : X (batch_num, 1, w,h)
#     output : Y (batch_num, num, w,h)
#     '''
#     def __init__(self, incoming, filter_size, stride=1, pad=0, flip_filters=False,  **kwargs):
#         super(LocalExpandLayer, self).__init__(incoming, **kwargs)
#
#         self.filter_size = lasagne.utils.as_tuple(filter_size, 2, int)
#         self.stride = lasagne.utils.as_tuple(stride, 2, int)
#         self.pad = lasagne.utils.as_tuple(pad, 2, int)
#         self.flip_filters = flip_filters
#
#     def get_output_shape_for(self, input_shape):
#         shape = (input_shape[0], np.prod(self.filter_size), input_shape[2], input_shape[3])
#         return shape
#
#     def get_output_for(self, input, **kwargs):
#         conv_mode = 'conv' if self.flip_filters else 'cross'
#         border_mode = self.pad
#         if border_mode == 'same':
#             border_mode = tuple(s // 2 for s in self.filter_size)
#         filter_size = self.filter_size
#         filter_localexpand_np = np.reshape(np.eye(np.prod(filter_size),np.prod(filter_size)), (np.prod(filter_size), 1, filter_size[0],filter_size[1]))
#         filter_localexpand = T.cast(theano.shared(filter_localexpand_np), 'floatX')
#         input_localexpanded = conved = dnn.dnn_conv(img=input, kerns=filter_localexpand, subsample=self.stride, border_mode=border_mode, conv_mode=conv_mode)
#
#         return input_localexpanded

class DynamicFilterLayer(MergeLayer):
    def __init__(self, incomings, filter_size, stride=1, pad=0, flip_filters=False, grouping=False, **kwargs):
        super(DynamicFilterLayer, self).__init__(incomings, **kwargs)

        self.filter_size = lasagne.utils.as_tuple(filter_size, 3, int)
        self.stride = lasagne.utils.as_tuple(stride, 2, int)
        self.pad = lasagne.utils.as_tuple(pad, 2, int)
        self.flip_filters = flip_filters
        self.grouping = grouping

        if self.grouping:
            assert(filter_size[2] == 1)

    def get_output_shape_for(self, input_shapes):
        if self.grouping:
            shape = (input_shapes[0][0], input_shapes[0][1], input_shapes[0][2], input_shapes[0][3])
        else:
            shape = (input_shapes[0][0], 1, input_shapes[0][2], input_shapes[0][3])
        return shape

    def get_output_for(self, input, **kwargs):
        image = input[0]
        filters = input[1]

        conv_mode = 'conv' if self.flip_filters else 'cross'
        border_mode = self.pad
        if border_mode == 'same':
            border_mode = tuple(s // 2 for s in self.filter_size)
        filter_size = self.filter_size

        if self.grouping:
            filter_localexpand_np = np.reshape(np.eye(np.prod(filter_size), np.prod(filter_size)), (np.prod(filter_size), 1, filter_size[0], filter_size[1]))
            filter_localexpand = T.cast(theano.shared(filter_localexpand_np), 'floatX')

            outputs = []
            for i in range(3):
                input_localexpanded = dnn.dnn_conv(img=image[:,[i],:,:], kerns=filter_localexpand, subsample=self.stride, border_mode=border_mode, conv_mode=conv_mode)
                output = T.sum(input_localexpanded * filters, axis=1, keepdims=True)
                outputs.append(output)

            output = T.concatenate(outputs, axis=1)
        else:
            filter_localexpand_np = np.reshape(np.eye(np.prod(filter_size), np.prod(filter_size)), (np.prod(filter_size), filter_size[2], filter_size[0], filter_size[1]))
            filter_localexpand = T.cast(theano.shared(filter_localexpand_np), 'floatX')
            input_localexpanded = dnn.dnn_conv(img=image, kerns=filter_localexpand, subsample=self.stride, border_mode=border_mode, conv_mode=conv_mode)
            output = input_localexpanded * filters
            output = T.sum(output, axis=1, keepdims=True)

        return output