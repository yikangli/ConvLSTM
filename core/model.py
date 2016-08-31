import theano
import theano.tensor as tensor
from theano.tensor.extra_ops import fill_diagonal
import numpy as np
from theano import config
from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from utils import ortho_weight, norm_weight, xavier_weight, tanh, l2norm, numpy_floatX
from fclayer import *
from conv_lstm import *

import pdb

def get_layer(name):
    """
    Return param init and feedforward functions for the given layer name
    """
    layers = {'conv_lstm': ('param_init_conv_lstm', 'conv_lstm_layer'),
              'fc':('fc_init','fully_layer') }

    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))

def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # conv-LSTM
    params = get_layer('conv_lstm')[0](options, params)
    # fc
    params = get_layer('fc')[0](options, params)

    return params


def build_model(tparams, options):
    trng = RandomStreams(1234)
    b_size = options['batch']

    # tensor variable for video clips, appearance and optical flow
    y = tensor.vector('y',dtype='int8')
    x = tensor.tensor4('x', dtype=config.floatX)
    mask = tensor.matrix('mask', dtype=config.floatX)
   
    # conv-lstm embedding
    embx = get_layer('conv_lstm')[1](tparams,x,options,mask)
    embx = (embx * mask[:, :, None]).sum(axis=0)
    embx_n = embx / mask.sum(axis=0)[:, None]

    # fc for recognition task
    prob = get_layer('fc')[1](tparams,x,options,mask)
    pred = np.argmax(prob)
    cost = -tensor.log(prob[y]+1e-8)

    return trng, x, mask, cost, pred

