

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