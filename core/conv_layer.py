import theano
import numpy as np
import theano.tensor as T
from theano.tensor.signal.downsample import max_pool_2d
import numbers

def param_init_conv(options, params, prefix='conv', filter_shape, stride=(1,1), poolsize=(1, 1)):
    """
    Affine transformation + point-wise nonlinearity
    """
    if nin == None:
        nin = options['dim_proj']
    if nout == None:
        nout = options['nCategories']

    # there are "num input feature maps * filter height * filter width"
    # inputs to each hidden unit
    fan_in = np.prod(filter_shape[1:])
    # each unit in the lower layer receives a gradient from:
    # "num output feature maps * filter height * filter width" / pooling size
    fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) // np.prod(poolsize))
    params[prefix+'_w'] = theano.shared(value=xavier_weight(fan_in, fan_out), borrow=True)
    params[prefix+'_b'] = theano.shared(value=np.zeros((filter_shape[0],), dtype=theano.config.floatX), borrow=true)

    return params

def conv(rng, input, filter_shape, image_shape, stride=(1,1), poolsize=(1, 1), weights_path=None):
    assert image_shape[1] == filter_shape[1]
    
    conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            border_mode = 'valid',
            subsample=stride,
            input_shape=image_shape
        )
    output = relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
    params = [W_values, b_values]

    if weights_path:
        model.load_weights(weights_path)

    return params, output