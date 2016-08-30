from __future__ import print_function
__docformat__ = 'restructedtext en'

import os
import sys
import timeit

import numpy
import pdb
from os import listdir
from os.path import isfile, join

def makeDataList(path):
    vidname = [f for f in listdir(path) if isfile(join(path,f))]
     

def load_data():

    train_dataset = {}
    train_dataset['name'] = []
    train_dataset['target'] = []
    train_dataset['im'] = []

    valid_dataset = {}
    valid_dataset['name'] = []
    valid_dataset['target'] = []
    valid_dataset['im'] = []
 
    path = '../../UCF51/Frame/'

    vidname = [f for f in listdir(path) if isfile(join(path,f))]
    pdb.set_trace() 

    for i in vidname:

        print (vidname[i])


    return train_dataset, valid_dataset

def get_data(app_feats, of_feats, captions, mask_caps, batch_idx, caps_idx, batch_size):
    """
    Used to shuffle the dataset at each iteration.
    """
    new_video_idx = caps_idx[batch_idx:batch_idx + batch_size]/20

    video_feats = []
    optic_feats = []
    target_index = []
    for i in range(batch_size):
        video_feats.append(app_feats[new_video_idx[i]].T)
        optic_feats.append(of_feats[new_video_idx[i]].T)

    max_aplen = 0
    max_oplen = 0
    for j in range(batch_size):
        max_aplen = max(max_aplen,video_feats[j].shape[0])
        max_oplen = max(max_oplen,optic_feats[j].shape[0])

    v_feats = numpy.zeros((max_aplen,batch_size,4096))
    v_optic = numpy.zeros((max_oplen,batch_size,4096))
    v_mask_app = numpy.zeros((max_aplen,batch_size))
    v_mask_of  = numpy.zeros((max_oplen,batch_size))
    v_caps = captions[:,caps_idx[batch_idx:batch_idx + batch_size]]
    v_mask_caps = mask_caps[:,caps_idx[batch_idx:batch_idx + batch_size]]

    for j in range(batch_size):
        v_feats[0:video_feats[j].shape[0],j,:] = video_feats[j]
        v_optic[0:optic_feats[j].shape[0],j,:] = optic_feats[j]
        v_mask_app[0:video_feats[j].shape[0],j]= 1
        v_mask_of[0:optic_feats[j].shape[0],j] = 1

    return v_feats, v_optic, v_mask_app, v_mask_of, v_caps, v_mask_caps  

def get_caps_data(captions,worddict,maxlen,n_words):
    seqs = []
    feat_list = []
    for i, cc in enumerate(captions):
        seqs.append([worddict[w] if worddict[w] < n_words else 1 for w in cc.lower().split()])

    lengths = [len(s) for s in seqs]

    if maxlen != None and numpy.max(lengths) >= maxlen:
        new_seqs = []
        new_lengths = []
        for l, s in zip(lengths, seqs):
            if l < maxlen:
                new_seqs.append(s)
                new_lengths.append(l)
        lengths = new_lengths
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)+1

    x = numpy.zeros((maxlen,n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen,n_samples)).astype('float32')
    for idx, s in enumerate(seqs):
        x[:lengths[idx],idx] = s
        x_mask[:lengths[idx]+1,idx] = 1.

    return x, x_mask
