def dfn(inputs,
         filters,
         kernel_size,
         stride=1,
         padding='SAME',
         scope=None,
         reuse=None):

  with tf.variable_op_scope([inputs, filters], scope, 'DFN', reuse=reuse):
    kernel_h, kernel_w = _two_element_tuple(kernel_size)
    stride_h, stride_w = _two_element_tuple(stride)
    num_filters_in = inputs.get_shape()[-1].value
    assert num_filters_in <= 3

    filter_size_prod = kernel_h * kernel_w * 1
    reshape_filters = np.reshape(
      np.eye(filter_size_prod, filter_size_prod),
      (kernel_h, kernel_w, 1, filter_size_prod)
    )
    reshape_filters = tf.constant(reshape_filters, dtype=tf.float32)

    outputs = []
    for i in range(num_filters_in):
      inputs_channel = tf.slice(inputs, [0,0,0,i], [-1,-1,-1,1])
      inputs_expanded = tf.nn.conv2d(inputs_channel, reshape_filters, [1, stride_h, stride_w, 1], padding=padding)
      output = tf.mul(inputs_expanded, filters)
      output = tf.reduce_sum(output, reduction_indices=-1, keep_dims=True)
      outputs.append(output)

    outputs = tf.concat(3, outputs)

    return outputs