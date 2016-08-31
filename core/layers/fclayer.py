import theano
import theano.tensor as tensor
from theano import config
from theano.tensor.nnet import conv2d
from theano.tensor.nnet import relu
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from utils import ortho_weight, norm_weight, xavier_weight, tanh, linear, numpy_floatX
import pdb 
import cv2

def param_init_fclayer(options, params, prefix='fc', nin=None, nout=None, ortho=True):
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

def fc_init(options, params, nCategories=101,imshape=None):
    """
    Affine transformation + point-wise nonlinearity
    """
    params = param_init_fclayer(options, params, prefix='fc1', nin = 384, nout = 512)
    params = param_init_fclayer(options, params, prefix='fc2', nin = 512, nout = nCategories)
  
    return params

def fully_layer(params, input, results, nCategories=101, nout=512, weights_path=None):
    trng = RandomStreams(SEED)
    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))
    ninput = tensor.prod(input.shape[1:])
    denselayer1 = tensor.dot(input, params['fc1_w']) + params['fc1_b']
    denselayer1 = relu(denselayer1)
    denselayer2 = tensor.dot(denselayer1, params['fc2_w']) + params['fc2_b']
    denselayer2 = relu(denselayer2)
    results['fc1'] = denselayer1
    results['fc2'] = denselayer2

    return params, results