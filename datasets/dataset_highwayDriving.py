# adapted from https://github.com/emansim/unsupervised-videos
# DataHandler for different types of datasets

from __future__ import division

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class DataHandler(object):
  """Data Handler that creates Bouncing MNIST dataset on the fly."""
  def __init__(self, image_size=64, binarize=False, num_frames=20, batch_size=2, mode='train'):
    if mode == 'train':
      dataset = 'datasets/highwayDriving_train.h5'
    elif mode == 'test':
      dataset = 'datasets/highwayDriving_test.h5'

    self.image_size_ = image_size
    self.binarize_ = binarize
    self.seq_length_ = num_frames
    self.batch_size_ = batch_size

    try:
      f = h5py.File(dataset)
    except:
      print 'Please set the correct path to MNIST dataset'
      sys.exit()
      
    # self.data_ = f['train'].value.reshape(-1, 28, 28)
    # import pdb; pdb.set_trace()
    self.data_ = f['highway_L'].value.transpose(0,1,3,2)
    self.dataset_size_ = self.data_.shape[0]
    self.num_channels_ = self.data_.shape[1]
    self.image_size_ = self.data_.shape[2]
    self.frame_size_ = self.image_size_ ** 2

  def GetBatchSize(self):
    return self.batch_size_

  def GetDims(self):
    return self.frame_size_

  def GetDatasetSize(self):
    return self.dataset_size_ - self.seq_length_

  def GetSeqLength(self):
    return self.seq_length_

  def Reset(self):
    pass

  def GetBatch(self, ind=None):
    if ind is None:
      batch_size = self.batch_size_
      ind = np.random.choice(self.dataset_size_ - self.seq_length_, batch_size)
    else:
      batch_size = len(ind)

    # import pdb; pdb.set_trace()

    # minibatch data
    data = np.ones((batch_size, self.num_channels_, self.image_size_, self.image_size_, self.seq_length_),
                  dtype=np.float32)
    
    for j in xrange(batch_size):
      # import pdb; pdb.set_trace()
      data[j, :, :, :, :] = self.data_[ind[j]:ind[j]+self.seq_length_, :, :, :].transpose(1,2,3,0)

    return data

  def DisplayData(self, data, rec=None, fut=None, fig=1, case_id=0, output_file=None):
    output_file1 = None
    output_file2 = None
    
    if output_file is not None:
      name, ext = os.path.splitext(output_file)
      output_file1 = '%s_original%s' % (name, ext)
      output_file2 = '%s_recon%s' % (name, ext)
    
    # get data
    data = data[case_id, :].reshape(-1, self.image_size_, self.image_size_, self.num_channels_)
    # get reconstruction and future sequences if exist
    if rec is not None:
      rec = rec[case_id, :].reshape(-1, self.image_size_, self.image_size_, self.num_channels_)
      enc_seq_length = rec.shape[0]
    if fut is not None:
      fut = fut[case_id, :].reshape(-1, self.image_size_, self.image_size_, self.num_channels_)
      if rec is None:
        enc_seq_length = self.seq_length_ - fut.shape[0]
      else:
        assert enc_seq_length == self.seq_length_ - fut.shape[0]
    
    num_rows = 1
    # create figure for original sequence
    plt.figure(2*fig, figsize=(10, 2))
    plt.clf()
    for i in xrange(self.seq_length_):
      plt.subplot(num_rows, self.seq_length_, i+1)
      plt.imshow(data[i, :, :, :].squeeze(), cmap=plt.cm.gray, interpolation="nearest")
      plt.axis('off')
    plt.show()
    if output_file1 is not None:
      print output_file1
      plt.savefig(output_file1, bbox_inches='tight')

    # create figure for reconstuction and future sequences
    plt.figure(2*fig+1, figsize=(10, 2))
    plt.clf()
    for i in xrange(self.seq_length_):
      if rec is not None and i < enc_seq_length:
        plt.subplot(num_rows, self.seq_length_, i + 1)
        plt.imshow(rec[rec.shape[0] - i - 1, :, :, :].squeeze(), cmap=plt.cm.gray, interpolation="nearest")
      if fut is not None and i >= enc_seq_length:
        plt.subplot(num_rows, self.seq_length_, i + 1)
        plt.imshow(fut[i - enc_seq_length, :, :, :].squeeze(), cmap=plt.cm.gray, interpolation="nearest")
      plt.axis('off')
    plt.show()
    if output_file2 is not None:
      print output_file2
      plt.savefig(output_file2, bbox_inches='tight')
    else:
      plt.pause(0.1)
