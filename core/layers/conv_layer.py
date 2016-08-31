import theano
import numpy as np
import theano.tensor as T
from theano.tensor.signal.downsample import max_pool_2d
import numbers
def param_init_conv(options, params, prefix='fc', nin=None, nout=None, ortho=True):
    """
    Affine transformation + point-wise nonlinearity
    """
    if nin == None:
        nin = options['dim_proj']
    if nout == None:
        nout = options['nCategories']
    params[prefix+'_w'] = theano.shared(value=xavier_weight(nin, nout), borrow=True)
    #params[_p(prefix,'_b')] = theano.shared(value=numpy.zeros((nout,)).astype('float32'), bowrrow=True)

    return params

def conv(rng, input, filter_shape, image_shape, stride=(1,1), poolsize=(2, 2), weights_path=None):
    assert image_shape[1] == filter_shape[1]

    # there are "num input feature maps * filter height * filter width"
    # inputs to each hidden unit
    fan_in = np.prod(filter_shape[1:])
    # each unit in the lower layer receives a gradient from:
    # "num output feature maps * filter height * filter width" /
    #   pooling size
    fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
               np.prod(poolsize))
    # initialize weights with random weights
    #W_bound = xavier_weight(fan_in, fan_out)
    W_bound = np.sqrt(6. / (fan_in + fan_out))
    W_values = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),dtype=theano.config.floatX), borrow=True)

    # the bias is a 1D tensor -- one bias per output feature map
    b_values = theano.shared(value=np.zeros((filter_shape[0],), dtype=theano.config.floatX), borrow=true)
    #b = theano.shared(value=b_values, borrow=True)
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