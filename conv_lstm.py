
from __future__ import print_function
import six.moves.cPickle as pickle

from collections import OrderedDict
import sys
import time

import numpy as np
import progressbar
import pdb
import theano
import gzip
import cPickle
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from load_data import *
from utils import *
from optim import *
from vocab import *
from model import init_params, build_model

# Set the random number generators' seeds for consistency
def train_lstm(
    imshape = (3,224,224),
    nCategories = 51,
    batch = 10,

    max_epochs=3,  # The maximum number of epoch to run.
    lrate = [0.001,0.001,0.001],
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    maxlen=100,  # Maximum length of captions 
    optimizer='rmsprop',  # sgd, adadelta and rmsprop available
    saveto='classification_model',  # The best model will be saved there.
    dispFreq=10000, # Display to stdout the training progress every N updates.
    saveFreq=20000, #9210,  # Save the parameters after every saveFreq updates.
    validFreq=140000, #2*9210,
    testFreq=5*140000,
    margin = 0.55,

    # Parameter for extra option.
    reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
    load_from_old = False,
    ):

    # Model options
    model_options = {}
    model_options['image_shape'] = imshape
    model_options['nCategories'] = nCategories
    
    model_options['batch'] = batch
    model_options['lrate'] = lrate
    model_options['max_epochs'] = max_epochs
    model_options['optimizer'] = optimizer

    model_options['dispFreq'] = dispFreq
    model_options['validFreq'] = validFreq
    model_options['saveto'] = saveto
    model_options['saveFreq'] = saveFreq
    
    ########################################################
    # Load training and development sets

    print('\nBuilding Forward Pass')
    # Read from Previous Training Result
    if load_from_old:
        print('Loading Previously Trained Weight')
        start = 14
        with open('../save/weights_14.pkl', 'rb') as W:
            tparams = pickle.load(W)
 
    else:
        start = 0
        params = init_params(model_options)
        tparams = init_tparams(params)
    pdb.set_trace()
    trng, use_noise, x, mask, cost, pred= build_model(params, model_options)
    print('Done')
    
    pdb.set_trace()

    print('\nBuilding Backward Pass')
    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay
    
    f_cost = theano.function(inputs = [mask,x,y],
                             outputs = [cost],
                             name='f_cost',
                             allow_input_downcast=True)

    grads = tensor.grad(cost, wrt=itemlist(tparams))
    f_grad = theano.function(inputs = [mask,x,y],
                             outputs = grads, 
                             name='f_grad',
                             allow_input_downcast=True)
    print('Done')
    ########################################################
    print('Optimization')
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, x, mask, cost)
    ########################################################
    '''
    print('Loading data...')
    model_options['validFreq'] = validFreq
    model_options['testFreq'] = testFreq
    train_dataset, valid_dataset = load_data(model_options)

    train_app_feats = train_dataset['App']
    train_of_feats = train_dataset['OF']
    train_target = train_dataset['target']
    train_clips = train_dataset['name']

    valid_app_feats = valid_dataset['App']
    valid_of_feats = valid_dataset['OF']
    valid_target = valid_dataset['target']
    valid_clips = valid_dataset['name']

    train_caps, train_caps_mask = get_caps_data(train_dataset['target'],worddict,maxlen,n_words)
#    valid_caps, valid_caps_mask = get_caps_data(valid_dataset['target'],worddict,maxlen,n_words)

    train_caps_length = train_caps.shape[1]
#    valid_caps_length = valid_caps.shape[1]
    print('Done')

   
    print("%d train examples" % len(train_app_feats))
    pdb.set_trace()
    #print("%d test examples" % len(test_labels))
    #print("%d valid examples" % len(valid_app_feats)) 

    history_errs = []
    best_p = None
    bad_count = 0

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()

    try:
        for eidx in range(start,max_epochs):
            n_samples = 0
            cost = 0
            show_cost = 0
            total_cost_sep = numpy.zeros((4,))
            # Get new index for shuffled cpations
            new_cap_idx = numpy.random.permutation(train_caps_length)
 
            bar = progressbar.ProgressBar(maxval = 1120,  \
                                        widgets = [progressbar.AnimatedMarker(), \
                                        'epoch progress', \
                                        ' ' , progressbar.Counter(),\
                                        ' ' , progressbar.Percentage(),\
                                        ' ', progressbar.ETA(), \
                                        ]).start()

            for i in range(0,140000,batch_size):
                uidx += batch_size
                use_noise.set_value(1.0)
                # Select the one video clip to train
                v_feats, v_optic, v_mask_app, v_mask_of, v_caps, v_mask_caps = get_data(train_app_feats, train_of_feats, train_caps, train_caps_mask, i, new_cap_idx, batch_size)
 
                n_samples += 1
                cost= f_grad_shared(v_feats, v_mask_app, v_optic, v_mask_of, v_caps, v_mask_caps)
                f_update(lrate[eidx])
                total_cost = f_cost(v_feats, v_mask_app, v_optic, v_mask_of, v_caps, v_mask_caps)
                total_cost_sep[0] += total_cost[0] #Total
                total_cost_sep[1] += total_cost[1] #cost_s
                total_cost_sep[2] += total_cost[2] #cost_v
                total_cost_sep[3] += total_cost[3] #cost_cap

                bar.update(n_samples)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if numpy.mod(uidx,dispFreq) == 0:
                    print('Epoch ', eidx, 'lrate  ',lrate[eidx], 'Update ', uidx, 'Cost ',total_cost_sep[0])
                    print ("Cap-Video cost: " + str(total_cost_sep[1]) + " Video-Cap cost: " + str(total_cost_sep[2]) + " Cap cost: " + str(total_cost_sep[3]))
                    total_cost_sep = numpy.zeros((4,))           

                if numpy.mod(uidx, saveFreq) == 0:
                    pickle.dump(model_options, open('../save/model.pkl','wb'),-1)
                    pickle.dump(tparams, open('../save/weights_%d.pkl' % eidx,'wb',-1))

            bar.finish()
            print('Seen %d batches' % n_samples)

            if estop:
                break 

    except KeyboardInterrupt:
        print("Training interupted")

    end_time = time.time()
    print('The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
    print( ('Training took %.1fs' %
            (end_time - start_time)), file=sys.stderr)

'''

if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    import conv_lstm
    conv_lstm.train_lstm()

