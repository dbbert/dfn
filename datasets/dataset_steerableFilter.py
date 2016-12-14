# adapted from https://github.com/emansim/unsupervised-videos
# DataHandler for different types of datasets

from __future__ import division

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scipy.signal import convolve2d

class DataHandler(object):
  """Data Handler that creates Bouncing MNIST dataset on the fly."""
  def __init__(self, dataset='datasets/movingObjects.h5', mode='standard', image_size=64, num_frames=3, batch_size=32):
    self.image_size_ = image_size
    self.seq_length_ = num_frames
    self.num_channels_ = 1
    self.mode_ = mode
    self.batch_size_ = batch_size

    try:
      f = h5py.File(dataset)
    except:
      print 'Please set the correct path to MNIST dataset'
      sys.exit()

    self.backgroundData_ = f['backgrounds'].value.transpose(0,1,3,2)
    self.num_backgrounds_ = self.backgroundData_.shape[0]
    f.close()

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

  def GetBatch(self, thetas=None, filter_size=9, sigma=1, batch_size_=None):

    if batch_size_ is None:
        batch_size_ = self.batch_size_

    # import pdb; pdb.set_trace()
    # images = np.random.rand(batch_size_, self.num_channels_, self.image_size_, self.image_size_, 1)
    if self.mode_ == 'standard':
      ind = np.random.choice(self.num_backgrounds_, batch_size_)
      images = self.backgroundData_[ind, ..., np.newaxis]
    elif self.mode_ == 'random':
      images = np.random.rand(batch_size_, self.num_channels_, self.image_size_, self.image_size_, 1)
    else:
      assert false

    if thetas is None:
      thetas = np.random.rand(batch_size_) * 2 * np.pi
    thetasChannel = np.ones((batch_size_, 1, self.image_size_, self.image_size_, 1)) * thetas[..., np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    # import pdb; pdb.set_trace()

    filteredImages = np.zeros((batch_size_, self.num_channels_, self.image_size_, self.image_size_, 1))
    for i in range(batch_size_):
      image = images[i].squeeze()
      theta = thetas[i]
      filteredImage = self.FilterWithTheta(image, theta, sigma, filter_size)
      # import pdb; pdb.set_trace()
      filteredImages[i, :, :, :, :] = filteredImage[None, None, :, :, None]

    output = np.concatenate((images, thetasChannel, filteredImages), axis=4)
    return output

  def FilterWithTheta(self, image, theta, sigma, filter_size):
    # https://www.mathworks.com/matlabcentral/fileexchange/9645-steerable-gaussian-filters/content/steerGauss.m
    # Evaluate 1D Gaussian filter( and its derivative).
    x = np.arange(-filter_size//2+1, filter_size//2+1)
    # import pdb; pdb.set_trace()
    g = np.array([np.exp(-(x**2) / (2*sigma**2))])
    gp = np.array([-(x / sigma) * np.exp(-(x**2) / (2*sigma**2))])

    Ix = convolve2d(image, -gp, mode='same', boundary='fill', fillvalue=0)
    Ix = convolve2d(Ix, g.T, mode='same', boundary='fill', fillvalue=0)

    Iy = convolve2d(image, g, mode='same', boundary='fill', fillvalue=0)
    Iy = convolve2d(Iy, -gp.T, mode='same', boundary='fill', fillvalue=0)

    output = np.cos(theta) * Ix + np.sin(theta) * Iy
    return output
