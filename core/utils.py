"""
Helper functions for multimodal-ranking
"""
import theano
import theano.tensor as tensor
import numpy

from collections import OrderedDict
from theano import config


def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params

def itemlist(tparams):
    """
    Get the list of parameters. 
    Note that tparams must be OrderedDict
    """
    return [vv for kk, vv in tparams.iteritems()]

def norm_weight(nin,nout=None, scale=0.1, ortho=True):
    """
    Uniform initalization from [-scale, scale]
    If matrix is square and ortho=True, use ortho instead
    """
    if nout == None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = numpy.random.uniform(low=-scale, high=scale, size=(nin, nout))
    return W.astype('float32')


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def xavier_weight(nin,nout=None):
    """
    Xavier init
    """
    if nout == None:
        nout = nin
    r = numpy.sqrt(6.) / numpy.sqrt(nin + nout)
    W = numpy.random.rand(nin, nout) * 2 * r - r
    return W.astype('float32')

def tanh(x):
    """
    Tanh activation function
    """
    return tensor.tanh(x)

def linear(x):
    """
    Linear activation function
    """
    return x

def ReLU(x):
    return tensor.nnet.relu(x)

def l2norm(X):
    """
    Compute L2 norm, row-wise
    """
    if X.ndim == 2:
        norm = tensor.sqrt(tensor.pow(X, 2).sum(1))
        X /= norm[:, None]
    elif X.ndim == 3:
	norm = tensor.sqrt(tensor.pow(X, 2).sum(2))
	X /= norm[:,:,None]
    else:
        norm = tensor.sqrt(tensor.pow(X, 2).sum())
        X /= norm
    
    return X

def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out



