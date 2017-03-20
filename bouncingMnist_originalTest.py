# adapted from https://github.com/emansim/unsupervised-videos
# DataHandler for different types of datasets

from __future__ import division

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class DataHandler(object):
  """Data Handler that creates Bouncing MNIST dataset on the fly."""
  def __init__(self, dataset='/path/to/bouncing_mnist_test.npy'):
    self.data_ = np.load(dataset)[..., np.newaxis].transpose(0,4,2,3,1)
    self.data_ = self.data_.astype(np.float32) / 255

    self.dataset_size_ = self.data_.shape[0]
    self.num_channels_ = self.data_.shape[1]
    self.image_size_ = self.data_.shape[2]
    self.frame_size_ = self.image_size_ ** 2

  def GetBatchSize(self):
    return self.batch_size_

  def GetDims(self):
    return self.frame_size_

  def GetDatasetSize(self):
    return self.dataset_size_

  def GetSeqLength(self):
    return self.seq_length_

  def Reset(self):
    pass

  def GetBatch(self, ind):
    # minibatch data
    data = self.data_[ind]

    return data
