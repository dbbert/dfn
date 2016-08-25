# adapted from https://github.com/emansim/unsupervised-videos

from __future__ import division

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class DataHandler(object):
  """Data Handler that creates Bouncing MNIST dataset on the fly."""
  def __init__(self, mnistDataset='datasets/mnist.h5', mode='standard', background='zeros', num_frames=20, batch_size=2, image_size=64, num_digits=2, step_length=0.1):
    self.mode_ = mode
    self.background_ = background
    self.seq_length_ = num_frames
    self.batch_size_ = batch_size
    self.image_size_ = image_size
    self.num_digits_ = num_digits
    self.step_length_ = step_length
    self.dataset_size_ = 10000  # The dataset is really infinite. This is just for validation.
    self.digit_size_ = 28
    self.frame_size_ = self.image_size_ ** 2
    self.num_channels_ = 1

    try:
      f = h5py.File(mnistDataset)
    except:
      print 'Please set the correct path to MNIST dataset'
      sys.exit()
      
    self.data_ = f['train'].value.reshape(-1, 28, 28)
    f.close()
    # if self.binarize_:
    #     self.data_ = np.round(self.data_)
    self.indices_ = np.arange(self.data_.shape[0])
    self.row_ = 0
    np.random.shuffle(self.indices_)

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

  def GetRandomTrajectory(self, batch_size):
    length = self.seq_length_
    canvas_size = self.image_size_ - self.digit_size_

    # Initial position uniform random inside the box.
    y = np.random.rand(batch_size)
    x = np.random.rand(batch_size)

    # Choose a random velocity.
    theta = np.random.rand(batch_size) * 2 * np.pi
    v_y = np.sin(theta)
    v_x = np.cos(theta)

    start_y = np.zeros((length, batch_size))
    start_x = np.zeros((length, batch_size))
    for i in xrange(length):
      # Take a step along velocity.
      y += v_y * self.step_length_
      x += v_x * self.step_length_

      # Bounce off edges.
      for j in xrange(batch_size):
        if x[j] <= 0:
          x[j] = 0
          v_x[j] = -v_x[j]
        if x[j] >= 1.0:
          x[j] = 1.0
          v_x[j] = -v_x[j]
        if y[j] <= 0:
          y[j] = 0
          v_y[j] = -v_y[j]
        if y[j] >= 1.0:
          y[j] = 1.0
          v_y[j] = -v_y[j]
      start_y[i, :] = y
      start_x[i, :] = x

    # Scale to the size of the canvas.
    start_y = (canvas_size * start_y).astype(np.int32)
    start_x = (canvas_size * start_x).astype(np.int32)
    return start_y, start_x

  def Overlap(self, a, b):
    """ Put b on top of a."""
    return np.maximum(a, b)
    #return b

  def GetBatch(self, verbose=False):
    start_y, start_x = self.GetRandomTrajectory(self.batch_size_ * self.num_digits_)
    
    # minibatch data
    if self.background_ == 'zeros':
        data = np.zeros((self.batch_size_, self.num_channels_, self.image_size_, self.image_size_, self.seq_length_), dtype=np.float32)
    elif self.background_ == 'rand':
        data = np.random.rand(self.batch_size_, self.num_channels_, self.image_size_, self.image_size_, self.seq_length_)
    
    for j in xrange(self.batch_size_):
      for n in xrange(self.num_digits_):
       
        # get random digit from dataset
        ind = self.indices_[self.row_]
        self.row_ += 1
        if self.row_ == self.data_.shape[0]:
          self.row_ = 0
          np.random.shuffle(self.indices_)
        digit_image = self.data_[ind, :, :]
        digit_size = self.digit_size_
        
        if self.mode_ == 'squares':
            digit_size = np.random.randint(5,20)
            digit_image = np.ones((digit_size, digit_size), dtype=np.float32)
        
        # import pdb; pdb.set_trace()
        
        # generate video
        for i in xrange(self.seq_length_):
          top    = start_y[i, j * self.num_digits_ + n]
          left   = start_x[i, j * self.num_digits_ + n]
          bottom = top  + digit_size
          right  = left + digit_size
          data[j, :, top:bottom, left:right, i] = self.Overlap(data[j, :, top:bottom, left:right, i], digit_image)
    
    return data

  def DisplayData(self, data, rec=None, fut=None, fig=1, case_id=0, output_file=None):
    output_file1 = None
    output_file2 = None
    
    if output_file is not None:
      name, ext = os.path.splitext(output_file)
      output_file1 = '%s_original%s' % (name, ext)
      output_file2 = '%s_recon%s' % (name, ext)
    
    # get data
    data = data[case_id, :].reshape(-1, self.image_size_, self.image_size_)
    # get reconstruction and future sequences if exist
    if rec is not None:
      rec = rec[case_id, :].reshape(-1, self.image_size_, self.image_size_)
      enc_seq_length = rec.shape[0]
    if fut is not None:
      fut = fut[case_id, :].reshape(-1, self.image_size_, self.image_size_)
      if rec is None:
        enc_seq_length = self.seq_length_ - fut.shape[0]
      else:
        assert enc_seq_length == self.seq_length_ - fut.shape[0]
    
    num_rows = 1
    # create figure for original sequence
    plt.figure(2*fig, figsize=(20, 1))
    plt.clf()
    for i in xrange(self.seq_length_):
      plt.subplot(num_rows, self.seq_length_, i+1)
      plt.imshow(data[i, :, :], cmap=plt.cm.gray, interpolation="nearest")
      plt.axis('off')
    plt.show()
    if output_file1 is not None:
      print output_file1
      plt.savefig(output_file1, bbox_inches='tight')

    # create figure for reconstuction and future sequences
    plt.figure(2*fig+1, figsize=(20, 1))
    plt.clf()
    for i in xrange(self.seq_length_):
      if rec is not None and i < enc_seq_length:
        plt.subplot(num_rows, self.seq_length_, i + 1)
        plt.imshow(rec[rec.shape[0] - i - 1, :, :], cmap=plt.cm.gray, interpolation="nearest")
      if fut is not None and i >= enc_seq_length:
        plt.subplot(num_rows, self.seq_length_, i + 1)
        plt.imshow(fut[i - enc_seq_length, :, :], cmap=plt.cm.gray, interpolation="nearest")
      plt.axis('off')
    plt.show()
    if output_file2 is not None:
      print output_file2
      plt.savefig(output_file2, bbox_inches='tight')
    else:
      plt.pause(0.1)
