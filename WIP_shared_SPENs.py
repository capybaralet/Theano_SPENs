#!/usr/bin/env python

"""
WIP

Code runs!

TODO:
    baseline MLP

TODO: 
    clean-up code; make it terse and readable

Start by writing a brief description here.
            we could extend it is a few ways:
                stress-test the assumption that the dependence between the y's doesn't depend on x
                    ACTUALLY, we should just make a dataset where that is not the case, and use gating to introduce a dependence...
                allow multiple guesses
                try and predict something with constraints (e.g. knapsack)
                    for the knapsack, we can output something too full / empty, and just rm/add things with the greedy heuristic, to make it easier
                    ISSUE: we need an "encoding" of the problem to condition on
                        a simple approach would be to use an RNN to "read in" the items in some order, e.g. in greedy heuristic order
                    ISSUE: there are as many output dimensions as objects!!!


Remember to remove WIP when code is "completed" (i.e. being used to run experiments)
"""

# FIXME: I should init y as the output of x...
#   2015 would just learn that via pretraining
#   2017 (E2E) learns it via backprop

#from __future__ import print_function
import numpy 
np = numpy

import os
import sys

import theano
import theano.tensor as T

from keras.optimizers import SGD
from keras.metrics import binary_crossentropy

from sklearn.metrics import f1_score # y_true, y_pred

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=25) # TODO: currently this must evenly divide nex 
parser.add_argument('--dataset', type=str, default='synthetic', choices=['synthetic', 'flickr', ''])
parser.add_argument('--y_init', type=str, default='0s', choices=['0s', 'features'])
parser.add_argument('--l2', type=float, default=0.)
parser.add_argument('--model', type=str, default='spen', choices=['spen', 'mlp', ''])
parser.add_argument('--nex', type=int, default=1500) # num_examples
#parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'momentum', 'sgd'])
parser.add_argument('--num_epochs', type=int, default=100)
#parser.add_argument('--num_train_steps', type=int, default=1000)
parser.add_argument('--num_inner_steps', type=int, default=100)
# TODO: finish this 
parser.add_argument('--pretrain', type=str, default='none', choices=['none', 'features', 'features_global', 'features_global_joint'],
        help="features: pretrain the feature network to predict y; \
              features_global: pretraing the feature network, then clamp these and train the global network" )
#
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--save_dir', type=str, default="./")
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--verbose', type=int, default=1)
#locals().update(parser.parse_args().__dict__)


# ---------------------------------------------------------------
print "PARSE ARGS and SET-UP SAVING (save_path/exp_settings.txt)"

args = parser.parse_args()
args_dict = args.__dict__

# save_path = filename + PROVIDED parser arguments
flags = [flag.lstrip('--') for flag in sys.argv[1:]]
flags = [ff for ff in flags if not ff.startswith('save_dir')]
save_dir = args_dict.pop('save_dir')
save_path = os.path.join(save_dir, os.path.basename(__file__) + '___' + '_'.join(flags))
args_dict['save_path'] = save_path

if args_dict['save']:
    # make directory for results, save ALL parser arguments
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open (os.path.join(save_path,'exp_settings.txt'), 'w') as f:
        for key in sorted(args_dict):
            f.write(key+'\t'+str(args_dict[key])+'\n')
    print( save_path)
    #assert False

locals().update(args_dict)

l2 = np.float32(l2)

# ---------------------------------------------------------------
print "SET RANDOM SEED (TODO: rng vs. random.seed)"


if seed is not None:
    np.random.seed(seed)  # for reproducibility
    rng = numpy.random.RandomState(seed)
else:
    rng = numpy.random.RandomState(np.random.randint(2**32 - 1))



# ---------------------------------------------------------------
print "DEFINE FUNCTIONS"

# from http://www.iro.umontreal.ca/~memisevr/code/logreg.py
def onehot(x,numclasses=None):
    """ Convert integer encoding for class-labels (starting with 0 !)
        to one-hot encoding.
        The output is an array who's shape is the shape of the input array plus
        an extra dimension, containing the 'one-hot'-encoded labels.
    """
    if x.shape==():
        x = x[None]
    if numclasses is None:
        numclasses = x.max() + 1
    result = numpy.zeros(list(x.shape) + [numclasses], dtype="int")
    z = numpy.zeros(x.shape, dtype="int")
    for c in range(numclasses):
        z *= 0
        z[numpy.where(x==c)] = 1
        result[...,c] += z
    return result


def parameter(shape, name='unnamed', init='normal'):
    if init == 'normal':
        return theano.shared((.01 * np.random.normal(0,1,shape)).astype('float32'), name=name, broadcastable=[dim==1 for dim in shape])
    elif init == 'bias':
        return theano.shared((.01 * np.ones(shape)).astype('float32'), name=name, broadcastable=[dim==1 for dim in shape])
    elif init == 'scalar': # for annoying broadcasting errors
        return theano.shared(np.float32(.01), name=name)

def hard_tanh(inp):
    return inp * (inp > 0) + (1 - inp) * (inp > 1)

# ---------------------------------------------------------------
print  "GET DATA"
input_size = 67
num_labels = 16

# TODO: visualization
if dataset == 'synthetic':
    X = np.random.randn(nex, input_size).astype('float32')
    A = np.random.randn(input_size, num_labels).astype('float32')
    Z = np.dot(X, A) # (N, 16)
    Z4x4 = Z.reshape((-1, 4, 4))
    Y_true = onehot(np.argmax(Z4x4, axis=-1)).reshape((-1, 16)).astype('float32')
    # VALID SET
    Xv = np.random.randn(nex, input_size).astype('float32')
    Av = np.random.randn(input_size, num_labels).astype('float32')
    Zv = np.dot(X, A) # (N, 16)
    Z4x4v = Z.reshape((-1, 4, 4))
    Y_truev = onehot(np.argmax(Z4x4, axis=-1)).reshape((-1, 16)).astype('float32')


def shuffle_data():
    global X, Xv, Y_true, Y_truev
    shuffles = np.random.permutation(nex)
    X = X[shuffles]
    Y_true = Y_true[shuffles]
    shuffles = np.random.permutation(nex)
    Xv = Xv[shuffles]
    Y_truev = Y_truev[shuffles]

def get_l2(theano_vars):
    l2 = 0.
    for var in theano_vars:
        l2 += ((var.flatten() ** 2).sum())**.5
    return l2

# ---------------------------------------------------------------
print "SET-UP TRAINING"
# See sections 3, 7.3, A.4, A.5


x = T.matrix('float32')
x.tag.test_value = X[:batch_size]
# following Belanger et al. (2017), we optimize the logit instead of doing mirror descent
y_logit_shared = theano.shared(np.zeros((batch_size, num_labels)).astype('float32'), name='y_logit_shared')
y_logit = T.matrix('float32')
y_bar = T.nnet.sigmoid(y_logit)
y_true = T.matrix('float32')
y_true.tag.test_value = Y_true[:batch_size]



#---------------
print "DEFINE LOCAL ENERGY"

# feature net
local_params = []
feature_Ws = [parameter((input_size, num_labels), 'feature_W')]
local_params += feature_Ws
feature_bs = [parameter((num_labels,), 'feature_b', 'bias')]
local_params += feature_bs
features = T.dot(x, feature_Ws[0]) + feature_bs[0]

# local pretraining
local_pretraining_loss = binary_crossentropy(y_true, T.nnet.sigmoid(features)).mean() + l2 * get_l2(local_params)
local_pretraining_preds = T.cast(T.nnet.sigmoid(features)>.5, 'float32')
local_pretraining_error = T.neq(y_true, local_pretraining_preds).mean() 
local_pretraining_step = theano.function( inputs=[x, y_true], 
                                      outputs=[local_pretraining_loss, local_pretraining_error, local_pretraining_preds],
                                      updates=SGD(lr=.01, momentum=.9).get_updates(local_params, [local_pretraining_loss, local_pretraining_error], local_pretraining_loss) )
# evaluation (no updates, i.e. no learning)
local_pretraining_stepv = theano.function( inputs=[x, y_true], 
                                      outputs=[local_pretraining_loss, local_pretraining_error, local_pretraining_preds])
# this is used to initialize y_logits (section?)
get_features = theano.function( inputs=[x], outputs=features)

# In pretraining, we minimize cost when: (features > 0 AND y == 1) OR (features < 0 AND y == 0)
# To make these have LOW energy, we set the local energy to the NEGATIVE pairwise product of features and y
local_energy = lambda y : T.sum(-y * features, axis=-1)


#---------------
print "DEFINE GLOBAL ENERGY"
global_Ws = []
global_bs = []
global_Ws += [parameter((num_labels, 4), 'global_W0')]
global_bs += [parameter((4,), 'global_b0', 'bias')]
#global_Ws += [parameter((4, 1), 'global_W1')]
global_Ws += [parameter((4,), 'global_W1')]
global_bs += [parameter((1,), 'global_b1', 'scalar')]
global_params = global_Ws + global_bs
def global_energy(y):
    rval = hard_tanh(T.dot(y, global_Ws[0]) + global_bs[0])
    return T.dot(rval, global_Ws[1]) + global_bs[1]

params = local_params + global_params


energy = lambda y : local_energy(y) + global_energy(y)
energy_y_bar = energy(y_bar)
energy_y_true = energy(y_true)


#---------------
print "DEFINE SSVM TRAINING"

y_pred = T.cast(y_bar > .5, 'float32')
zero_one_loss = T.neq(y_pred, y_true).mean(-1)
# we approximate the 0-1 loss with log-loss (Section 4, 2nd column)
surrogate_loss = binary_crossentropy(y_true, y_bar)


# loss-augmented inference
y_opt = SGD(lr=.1, momentum=.95)
y_step_outputs = [y_bar, y_pred]
# FIXME: these need to depend on y_logit_shared...
y_step = theano.function([x, y_true], y_step_outputs, updates=y_opt.get_updates([y_logit_shared], [], (surrogate_loss + energy_y_bar).mean()))
# at test time, we don't get to see the ground truth, we just find the y which minimizes the energy
y_stepv = theano.function([x], y_step_outputs, updates=y_opt.get_updates([y_logit_shared], [], (energy_y_bar).mean()))

# FIXME: train_loss = nan?

# SSVM loss; energy_y is **approximately** minimized by iterative inference (the y_step function above)
train_loss = surrogate_loss + energy_y_true - energy_y_bar
train_loss = train_loss * (train_loss > 0)
#global_train_loss = train_loss + l2 * get_l2(global_params)
train_loss = train_loss * (train_loss > 0) + get_l2(params)
train_updates = SGD(lr=.01, momentum=.9).get_updates(params, [], train_loss.mean())
train_step_outputs = [train_loss, surrogate_loss, energy_y_true, energy_y_bar, y_true, y_pred, zero_one_loss]
train_step = theano.function([x, y_true], train_step_outputs, updates=train_updates, givens={y_logit: y_logit_shared})

# for the global training, we only update the global params
global_train_updates = SGD(lr=.01, momentum=.9).get_updates(global_params, [], train_loss.mean())
global_train_step = theano.function([x, y_true], train_step_outputs, updates=train_updates, givens={y_logit: y_logit_shared})

# evaluation (no updates, i.e. no learning)
train_stepv = theano.function([x, y_true], train_step_outputs, givens={y_logit: y_logit_shared})

#TODO mlp_baseline = 64, 64, 16, 16


# ---------------------------------------------------------------
print "TRAIN (and SAVE)"

"""
Training has 3 phases (see section A.4 (2015)):
    1. pretrain local energy net
    2. fix local enery net and train global energy net
    3. jointly fine tune both networks
"""

num_batches = nex / batch_size
num_train_steps = num_epochs * num_batches

# ----------
# MONITORING:
#   I track loss, error, and F1
local_pretraining_monitored  = np.empty((num_epochs, 3))
local_pretraining_monitoredv = np.empty((num_epochs, 3))
global_monitored  = np.empty((num_epochs, 3))
global_monitoredv = np.empty((num_epochs, 3))
monitored  = np.empty((num_epochs, 3))
monitoredv = np.empty((num_epochs, 3))

# -------------------------
if 1:
    print "\n PRETRAINING local energy\n"
    for epoch in range(num_epochs):
        # TRAINING STEPS
        shuffle_data()
        for batch in range(num_batches): 
            XX = X[batch*batch_size: (batch+1)*batch_size]
            YY = Y_true[batch*batch_size: (batch+1)*batch_size]
            local_pretraining_step(XX, YY)

        # MONITORING
        # train
        loss, error, y_preds = local_pretraining_stepv(X, Y_true)
        local_pretraining_monitored[epoch] = loss, error, np.mean([f1_score(Y_true[nn], y_preds[nn], average='macro') for nn in range(nex)])
        # valid
        loss, error, y_preds = local_pretraining_stepv(Xv, Y_truev)
        local_pretraining_monitoredv[epoch] = loss, error, np.mean([f1_score(Y_truev[nn], y_preds[nn], average='macro') for nn in range(nex)])
        print "(train)  loss, error, F1:  ", local_pretraining_monitored[epoch],'           (valid):  ', local_pretraining_monitoredv[epoch]


    
if 1:
    print "\nPRETRAINING global energy\n"
    for epoch in range(num_epochs):
        # TRAINING STEPS
        shuffle_data()
        for batch in range(num_batches): 
            XX = X[batch*batch_size: (batch+1)*batch_size]
            YY = Y_true[batch*batch_size: (batch+1)*batch_size]
            # initialize y
            if y_init == '0s':
                y_logit_shared.set_value((.0 * np.ones((batch_size, num_labels))).astype('float32'))
            else:
                y_logit_shared.set_value(get_features(XX))
            # optimize y
            for inner_step in range(num_inner_steps):
                y_step(XX, YY)
            global_train_step(XX, YY)

        # MONITORING
        # train
        # initialize y
        if y_init == '0s':
            y_logit_shared.set_value((.0 * np.ones((batch_size, num_labels))).astype('float32'))
        else:
            y_logit_shared.set_value(get_features(X))
        # optimize y
        for inner_step in range(num_inner_steps):
            y_stepv(X)
        # outs = [train_loss, surrogate_loss, energy_y_true, energy_y_bar, y_true, y_pred, zero_one_loss]
        outs = train_stepv(X, Y_true)
        global_monitored[epoch] = outs[0].mean(), outs[-1].mean(), np.mean([f1_score(outs[-3][nn], outs[-2][nn], average='macro') for nn in range(batch_size)])

        # valid
        # initialize y
        if y_init == '0s':
            y_logit_shared.set_value((.0 * np.ones((batch_size, num_labels))).astype('float32'))
        else:
            y_logit_shared.set_value(get_features(Xv))
        # optimize y
        for inner_step in range(num_inner_steps):
            y_stepv(Xv)
        # outs = [train_loss, surrogate_loss, energy_y_true, energy_y_bar, y_true, y_pred, zero_one_loss]
        outs = train_stepv(Xv, Y_truev)
        global_monitoredv[epoch] = outs[0].mean(), outs[-1].mean(), np.mean([f1_score(outs[-3][nn], outs[-2][nn], average='macro') for nn in range(batch_size)])

        print "(train)  loss, error, F1:  ", global_monitored[epoch],'           (valid):  ', global_monitoredv[epoch]

    
if 1:
    print "\nJOINTLY TRAINING local and global energy\n"
    for epoch in range(num_epochs):
        # TRAINING STEPS
        shuffle_data()
        for batch in range(num_batches): 
            XX = X[batch*batch_size: (batch+1)*batch_size]
            YY = Y_true[batch*batch_size: (batch+1)*batch_size]
            # initialize y
            if y_init == '0s':
                y_logit_shared.set_value((.0 * np.ones((batch_size, num_labels))).astype('float32'))
            else:
                y_logit_shared.set_value(get_features(XX))
            # optimize y
            for inner_step in range(num_inner_steps):
                y_step(XX, YY)
            train_step(XX, YY)

        # MONITORING
        # train
        # initialize y
        if y_init == '0s':
            y_logit_shared.set_value((.0 * np.ones((batch_size, num_labels))).astype('float32'))
        else:
            y_logit_shared.set_value(get_features(X))
        # optimize y
        for inner_step in range(num_inner_steps):
            y_stepv(X)
        # outs = [train_loss, surrogate_loss, energy_y_true, energy_y_bar, y_true, y_pred, zero_one_loss]
        outs = train_stepv(X, Y_true)
        monitored[epoch] = outs[0].mean(), outs[-1].mean(), np.mean([f1_score(outs[-3][nn], outs[-2][nn], average='macro') for nn in range(batch_size)])

        # valid
        # initialize y
        if y_init == '0s':
            y_logit_shared.set_value((.0 * np.ones((batch_size, num_labels))).astype('float32'))
        else:
            y_logit_shared.set_value(get_features(Xv))
        # optimize y
        for inner_step in range(num_inner_steps):
            y_stepv(Xv)
        # outs = [train_loss, surrogate_loss, energy_y_true, energy_y_bar, y_true, y_pred, zero_one_loss]
        outs = train_stepv(Xv, Y_truev)
        monitoredv[epoch] = outs[0].mean(), outs[-1].mean(), np.mean([f1_score(outs[-3][nn], outs[-2][nn], average='macro') for nn in range(batch_size)])

        print "(train)  loss, error, F1:  ", monitored[epoch],'           (valid):  ', monitoredv[epoch]







# ---------------------------------------------------------------
# ANALYZE (e.g. PLOTS)