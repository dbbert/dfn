
# coding: utf-8
# export THEANO_FLAGS="device=gpu0, floatX=float32" optimizer=None

import theano
import theano.tensor as T
import numpy as np
import os
import socket
import argparse
import time
import datetime
import importlib

import matplotlib.pyplot as plt
from IPython import display

import lasagne
from lasagne.utils import floatX
from lasagne.updates import rmsprop, adam, momentum
from lasagne.layers import get_all_params, get_all_layers, get_all_param_values, get_output
from lasagne.objectives import squared_error, binary_crossentropy, aggregate

from utils.helperFunctions import *

def train(options):
    # -------- setup options and data ------------------
    np.random.seed(options['seed'])

    # Load options
    host = socket.gethostname() # get computer hostname
    start_time = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

    model = importlib.import_module(options['model_file'])

    # ---------- build model and compile ---------------
    input_batch = T.tensor4() # input image sequences
    target = T.tensor4() # target image

    print('Build model...')
    model = model.Model(**options['modelOptions'])

    print('Compile ...')
    net, outputs, filters = model.build_model(input_batch)

    # compute loss
    outputs = get_output(outputs + [filters])
    output_frames = outputs[:-1]
    output_filter = outputs[-1]

    train_losses = []
    for i in range(options['modelOptions']['target_seqlen']):
        output_frame = output_frames[i]

        if options['loss'] == 'squared_error':
            frame_loss = squared_error(output_frame, target[:, [i], :, :])
        elif options['loss'] == 'binary_crossentropy':
            # Clipping to avoid NaN's in binary crossentropy: https://github.com/Lasagne/Lasagne/issues/436
            output_frame = T.clip(output_frame, np.finfo(np.float32).eps, 1-np.finfo(np.float32).eps)
            frame_loss = binary_crossentropy(output_frame, target[:,[i],:,:])
        else:
            assert False

        train_losses.append(aggregate(frame_loss))

    train_loss = sum(train_losses) / options['modelOptions']['target_seqlen']

    # update
    sh_lr = theano.shared(lasagne.utils.floatX(options['learning_rate'])) # to allow dynamic learning rate

    layers = get_all_layers(net)
    all_params = get_all_params(layers, trainable = True)
    updates = adam(train_loss, all_params, learning_rate=sh_lr)
    _train = theano.function([input_batch, target], train_loss, updates=updates, allow_input_downcast=True)
    _test = theano.function([input_batch, target], [train_loss, output_filter] + output_frames, allow_input_downcast=True)

    # ------------ data setup ----------------
    print('Prepare data...')
    dataset = importlib.import_module(options['dataset_file'])
    dh = dataset.DataHandler(**options['datasetOptions'])

    # ------------ training setup ----------------
    if options['pretrained_model_path'] is not None:
        checkpoint = pickle.load(open(options['pretrained_model_path'], 'rb'))
        model_values = checkpoint['model_values'] # overwrite the values of model parameters
        lasagne.layers.set_all_param_values(layers, model_values)

        history_train = checkpoint['history_train']
        start_epoch = checkpoint['epoch'] + 1
        options['batch_size'] = checkpoint['options']['batch_size']
        sh_lr.set_value(floatX(checkpoint['options']['learning_rate']))
    else:
        start_epoch = 0
        history_train = []

    # ------------ actual training ----------------
    print 'Start training ...'

    input_seqlen = options['modelOptions']['input_seqlen']
    for epoch in range(start_epoch, start_epoch + options['num_epochs']):
        epoch_start_time = time.time()

        history_batch = []
        for batch_index in range(0, options['batches_per_epoch']):

            batch = dh.GetBatch() # generate data on the fly
            if options['dataset_file'] == 'datasets.stereoCarsColor':
                batch_input = batch[..., :input_seqlen].squeeze(axis=4)  # first frames
                batch_target = batch[..., input_seqlen:].squeeze(axis=4)  # last frame
            else:
                batch_input = batch[..., :input_seqlen].transpose(0,4,2,3,1).squeeze(axis=4) # first frames
                batch_target = batch[..., input_seqlen:].transpose(0,4,2,3,1).squeeze(axis=4) # last frame

            # train
            loss_train = _train(batch_input, batch_target)
            history_batch.append(loss_train)

            print("Epoch {} of {}, batch {} of {}, took {:.3f}s".format(epoch + 1, options['num_epochs'], batch_index+1, options['batches_per_epoch'], time.time() - epoch_start_time))
            print("  training loss:\t{:.6f}".format(loss_train.item()))

        # clear the screen
        display.clear_output(wait=True)

        # print statistics
        history_train.append(np.mean(history_batch))
        history_batch = []
        print("Epoch {} of {}, took {:.3f}s".format(epoch + 1, options['num_epochs'], time.time() - epoch_start_time))
        print("  training loss:\t{:.6f}".format(history_train[epoch].item()))

        # set new learning rate (maybe this is unnecessary with adam updates)
        if (epoch+1) % options['decay_after'] == 0:
            options['learning_rate'] = sh_lr.get_value() * 0.5
            print "New LR:", options['learning_rate']
            sh_lr.set_value(floatX(options['learning_rate']))

        # save the model
        if (epoch+1) % options['save_after'] == 0:
            save_model(layers, epoch, history_train, start_time, host, options)
            print("Model saved")