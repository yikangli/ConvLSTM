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
from utils import ortho_weight, norm_weight, xavier_weight, tanh, linear, numpy_floatX
import pdb 
import cv2

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

def get_layer(name):
    """
    Return param init and feedforward functions for the given layer name
    """
    layers = {'conv_lstm': ('param_init_conv_lstm', 'conv_lstm_layer'),
            'fc':('fc_init','fully_layer') }

    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))
