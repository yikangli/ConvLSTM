import theano
import theano.tensor as tensor
from theano import config
from theano.tensor.nnet import conv2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from utils import xavier_weight, numpy_floatX, ReLU
from conv_layer import *
import pdb 
import cv2

def inceptionv4(options,params,rng, input):
    results = OrderedDict()
    # inception v4 lower part
    params['W1_1'], results['layerR0'] = convolu_theano(rng, input,filter_shape=(32,3,3,3),image_shape=options['image_shape'],stride=(1,1),poolsize=(2,2), weights_path=None)
    params['W1_2'], results['layerR1'] = convolu_theano(rng, ReLU(results['layerR1']),filter_shape=(32,3,3,3),image_shape=layerR0.shape,stride=(0,0),poolsize=(2,2), weights_path=None)
    params['W1_3'], results['layerR2'] = convolu_theano(rng, ReLU(results['layerR2']),filter_shape=(64,3,3,3),image_shape=layerR1.shape,stride=(0,0),poolsize=(2,2), weights_path=None)
    params['W1_4'], results['layerR3'] = convolu_theano(rng, ReLU(results['layerR3']),filter_shape=(96,3,3,3),image_shape=layerR2.shape,stride=(1,1),poolsize=(2,2), weights_path=None)
    results['layerL1'] = fancy_max_pool(ReLU(results['layerR2']), pool_shape=(3,3), pool_stride=(1,1), ignore_border=False)
    # inception v4 lower part concatenation
    results['filter_concate_1'] = theano.concatenate([results['layerL1'], results['layerR3']], axis=3)

    # inception v4 middle part
    params['W2_1'], results['layerR4'] = convolu_theano(rng, results['filter_concate_1'], filter_shape=(64,3,1,1),image_shape=filter_concate_1.shape,stride=(0,0),poolsize=(2,2), weights_path=None)
    params['W2_2'], results['layerL2'] = convolu_theano(rng, results['filter_concate_1'], filter_shape=(64,3,1,1),image_shape=filter_concate_1.shape,stride=(0,0),poolsize=(2,2), weights_path=None)
    params['W2_3'], results['layerL3'] = convolu_theano(rng, ReLU(results['layerL2']), filter_shape=(96,3,1,1),image_shape=layerL2.shape,stride=(0,0),poolsize=(2,2), weights_path=None)
    params['W2_4'], results['layerR5'] = convolu_theano(rng, ReLU(results['layerR4']), filter_shape=(64,3,7,1),image_shape=layerR4.shape,stride=(0,0),poolsize=(2,2), weights_path=None)
    params['W2_5'], results['layerR6'] = convolu_theano(rng, ReLU(results['layerR5']), filter_shape=(64,3,1,7),image_shape=layerR5.shape,stride=(0,0),poolsize=(2,2), weights_path=None)
    params['W2_6'], results['layerR7'] = convolu_theano(rng, ReLU(results['layerR7']), filter_shape=(96,3,3,3),image_shape=layerR6.shape,stride=(0,0),poolsize=(2,2), weights_path=None)
    #inception v4 middle part concatenation
    results['filter_concate_2'] = theano.concatenate([results['layerL3'],results['layerR7'] ], axis=3)

    #inception v4 upper part
    results['layerR8'] = fancy_max_pool(results['filter_concate_2'], pool_shape=(2,2), pool_stride=(1,1), ignore_border=False)
    params['W3_1'], results['layerL4'] = convolu_theano(rng, results['filter_concate_2'] , filter_shape=(192,3,3,3),image_shape=filter_concate_2.shape, stride=(0,0), poolsize=(2,2), weights_path=None)
    #inception v4 upper part concatenation
    results['filter_concate_3'] = theano.concatenate([ReLU(results['layerL4']),results['layerR8']], axis=3)

    return params, results

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