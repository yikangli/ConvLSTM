"""
Layers for multimodal-ranking
"""
import theano
import theano.tensor as tensor
from theano import config
from theano.tensor.nnet import conv2d
from theano.tensor.nnet import relu
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import numpy
from utils import _p, ortho_weight, norm_weight, xavier_weight, tanh, linear, numpy_floatX
import pdb 
import cv2


class VGG16ConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, stride=(1,1), poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(input=input, 
            filters=self.W,
            filter_shape=filter_shape,
            subsample=stride,
            input_shape=image_shape,
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


def fancy_max_pool(input_tensor, pool_shape, pool_stride,
                    ignore_border=False):
    """Using theano built-in maxpooling, create a more flexible version.
    Obviously suboptimal, but gets the work done."""

    if isinstance(pool_shape, numbers.Number):
        pool_shape = pool_shape,
    if isinstance(pool_stride, numbers.Number):
        pool_stride = pool_stride,

    if len(pool_shape) == 1:
        pool_shape = pool_shape * 2
    if len(pool_stride) == 1:
        pool_stride = pool_stride * 2

    lcmh, lcmw = [_lcm(p, s) for p, s in zip(pool_shape, pool_stride)]
    dsh, dsw = lcmh / pool_shape[0], lcmw / pool_shape[1]

    pre_shape = input_tensor.shape[:-2]
    length = theano.prod(pre_shape)
    post_shape = input_tensor.shape[-2:]
    new_shape =theanoT.concatenate([[length], post_shape])
    reshaped_input = input_tensor.reshape(new_shape, ndim=3)
    sub_pools = []
    for sh in range(0, lcmh, pool_stride[0]):
        sub_pool = []
        sub_pools.append(sub_pool)
        for sw in range(0, lcmw, pool_stride[1]):
            full_pool = max_pool_2d(reshaped_input[:, sh:,
                                                      sw:],
                                    pool_shape, ignore_border=ignore_border)
            ds_pool = full_pool[:, ::dsh, ::dsw]
            concat_shape = T.concatenate([[length], ds_pool.shape[-2:]])
            sub_pool.append(ds_pool.reshape(concat_shape, ndim=3))
    output_shape = (length,
                    T.sum([l[0].shape[1] for l in sub_pools]),
                    T.sum([i.shape[2] for i in sub_pools[0]]))
    output = T.zeros(output_shape)
    for i, line in enumerate(sub_pools):
        for j, item in enumerate(line):
            output = T.set_subtensor(output[:, i::lcmh / pool_stride[0],
                                               j::lcmw / pool_stride[1]],
                                     item)
    return output.reshape(T.concatenate([pre_shape, output.shape[1:]]), ndim=input_tensor.ndim)


def convolu_theano(rng, input, filter_shape, image_shape, stride=(1,1), poolsize=(2, 2), weights_path=None):
    assert image_shape[1] == filter_shape[1]

    # there are "num input feature maps * filter height * filter width"
    # inputs to each hidden unit
    fan_in = numpy.prod(filter_shape[1:])
    # each unit in the lower layer receives a gradient from:
    # "num output feature maps * filter height * filter width" /
    #   pooling size
    fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
               numpy.prod(poolsize))
    # initialize weights with random weights
    #W_bound = xavier_weight(fan_in, fan_out)
    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    W_values = theano.shared(numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),dtype=theano.config.floatX), borrow=True)

    # the bias is a 1D tensor -- one bias per output feature map
    b_values = theano.shared(value=numpy.zeros((filter_shape[0],), dtype=theano.config.floatX), borrow=true)
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

def inceptionv4(options,params,rng, input):
    results = OrderedDict()
    # inception v4 lower part
    params['W1_1'], results['layerR0'] = convolu_theano(rng, input,filter_shape=(32,3,3,3),image_shape=options['image_shape'],stride=(1,1),poolsize=(2,2), weights_path=None)
    params['W1_2'], results['layerR1'] = convolu_theano(rng, layerR0,filter_shape=(32,3,3,3),image_shape=layerR0.shape,stride=(0,0),poolsize=(2,2), weights_path=None)
    params['W1_3'], results['layerR2'] = convolu_theano(rng, layerR1,filter_shape=(64,3,3,3),image_shape=layerR1.shape,stride=(0,0),poolsize=(2,2), weights_path=None)
    params['W1_4'], results['layerR3'] = convolu_theano(rng, layerR2,filter_shape=(96,3,3,3),image_shape=layerR2.shape,stride=(1,1),poolsize=(2,2), weights_path=None)
    results['layerL1'] = fancy_max_pool(layerR2, pool_shape=(3,3), pool_stride=(1,1), ignore_border=False)
    # inception v4 lower part concatenation
    results['filter_concate_1'] = theano.concatenate([layerL1, layerR3], axis=3)

    # inception v4 middle part
    params['W2_1'], results['layerR4'] = convolu_theano(rng, filter_concate_1, filter_shape=(64,3,1,1),image_shape=filter_concate_1.shape,stride=(0,0),poolsize=(2,2), weights_path=None)
    params['W2_2'], results['layerL2'] = convolu_theano(rng, filter_concate_1, filter_shape=(64,3,1,1),image_shape=filter_concate_1.shape,stride=(0,0),poolsize=(2,2), weights_path=None)
    params['W2_3'], results['layerL3'] = convolu_theano(rng, layerL2, filter_shape=(96,3,1,1),image_shape=layerL2.shape,stride=(0,0),poolsize=(2,2), weights_path=None)
    params['W2_4'], results['layerR5'] = convolu_theano(rng, layerR4, filter_shape=(64,3,7,1),image_shape=layerR4.shape,stride=(0,0),poolsize=(2,2), weights_path=None)
    params['W2_5'], results['layerR6'] = convolu_theano(rng, layerR5, filter_shape=(64,3,1,7),image_shape=layerR5.shape,stride=(0,0),poolsize=(2,2), weights_path=None)
    params['W2_6'], results['layerR7'] = convolu_theano(rng, layerR6, filter_shape=(96,3,3,3),image_shape=layerR6.shape,stride=(0,0),poolsize=(2,2), weights_path=None)
    #inception v4 middle part concatenation
    results['filter_concate_2'] = theano.concatenate([layerL3,layerR7], axis=3)

    #inception v4 upper part
    results['layerR8'] = fancy_max_pool(filter_concate_2, pool_shape=(2,2), pool_stride=(1,1), ignore_border=False)
    params['W3_1'], results['layerL4'] = convolu_theano(rng, filter_concate_2, filter_shape=(192,3,3,3),image_shape=filter_concate_2.shape, stride=(0,0), poolsize=(2,2), weights_path=None)
    #inception v4 upper part concatenation
    results['filter_concate_3'] = theano.concatenate([layerL4, layerR8], axis=3)

    return params, results


def param_init_fully(options, params, prefix='fc', nin=None, nout=None, ortho=True):
    """
    Affine transformation + point-wise nonlinearity
    """
    if nin == None:
        nin = options['dim_proj']
    if nout == None:
        nout = options['dim_proj']
    params[_p(prefix,'W')] = theano.shared(value=xavier_weight(nin, nout), borrow=True)
    params[_p(prefix,'b')] = theano.shared(value=numpy.zeros((nout,)).astype('float32'), bowrrow=True)

    return params


def dropout_layer(state_before, use_noise, trng):
    output = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return output


def fully_layer(params, input, results, nCategories=101, nout=1024, weights_path=None):
    trng = RandomStreams(SEED)
    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))
    ninput = tensor.prod(input.shape[1:])
    params = param_init_fully(options, params, prefix='fc_1', nin = ninput, nout = 1024)
    params = param_init_fully(options, params, prefix='fc_2', nin = 1024, nout = 1024)
    denselayer1 = tensor.dot(input, params['fc_1W']) + params['fc_1b']
    denselayer1 = relu(denselayer1)
    denselayer1 = dropout_layer(denselayer1, use_noise, trng)
    denselayer2 = tensor.dot(denselayer1, params['fc_2W']) + params['fc_2b']
    denselayer2 = relu(denselayer2)
    denselayer2 = dropout_layer(denselayer2, use_noise, trng)
    results['fc_1'] = denselayer1
    results['fc_2'] = denselayer2

    return params, results


def get_layer(name):
    """
    Return param init and feedforward functions for the given layer name
    """
    layers = {'conv_lstm': ('param_init_conv_lstm', 'conv_lstm_layer'),
            'fc':('param_init_fclayer','fclayer') }

    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))

def param_init_fclayer(options, params, nCategories=101,imshape=None):
    """
    Affine transformation + point-wise nonlinearity
    """
    params['fc_layer'] = fully_layer(None,nCategories,imshape)

    return params

def param_init_conv_lstm(options,params,channel=None, dim_h=None, dim_w=None):
    if channel == None:
        channel = options['channel']
    if dim_h == None:
        dim_h = options['dim_h']
    if dim_w == None:
        dim_w = options['dim_w']
    
    params['Wz'] = inceptionv4(None,channel,dim_h,dim_w);
    params['Uz'] = inceptionv4(None,channel,dim_h,dim_w);
    params['Wr'] = inceptionv4(None,channel,dim_h,dim_w);
    params['Ur'] = inceptionv4(None,channel,dim_h,dim_w);
    params['Wh'] = inceptionv4(None,channel,dim_h,dim_w);
    params['Uh'] = inceptionv4(None,channel,dim_h,dim_w);

    return params


#Skip-Thoughts
def conv_lstm_layer(tparams, state_below, options, mask):
    '''
    Feedforward pass through LSTM
    '''
    nsamples = state_below.shape[0]
    nsteps = state_below.shape[1]
    height = state_below.shape[2]
    width = state_below.shape[3]
    dim = height*width

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]
    pdb.set_trace()
    state_below_ = np.concatenate((tparams['Wz'].predict[state_below],tparams['Wr'].predict[state_below]),axis=1)
    state_belowx = tparams['Wh'].predict[state_below]

    def _step_slice(m_,x_, xx_, h_, U, Ux):
        preact = np.concatenate((tparams['Wz'].predict[h_],tparams['Wr'].predict[h_]),axis=1)
        preact += x_

        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        preactx = tparams['Wh'].predict[h_]
        preactx = preactx * r
        preactx = preactx + xx_

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:,None]*h+(1.-m_)[:,None]*h_

        return h

    seqs = [mask, state_below_, state_belowx]

    rval, updates = theano.scan(_step_slice,
                                sequences=seqs,
                                outputs_info = [None],
                                non_sequences = [tparams['Uz'],
                                                 tparams['Ur'],
                                                 tparams['Uh'],],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=False,
                                strict=True)
    rval = [rval]
    return rval[0]



