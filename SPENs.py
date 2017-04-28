#!/usr/bin/env python

"""
WIP

Code runs!

TODO:
    baseline MLP
    pretraining
    figure out energy "convergence"
    implement F1


Start by writing a brief description here.
    The plan is FIRST to implement SPENs and test on some *other* multi-classification dataset (they just did one, right??)
        * get working code
        * get data
        * run a few experiments
            baselines: DNN w/sigmoids, ...?
        * start writing
        * MORE??
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
from keras.metrics import categorical_crossentropy


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=25) # TODO: currently this must evenly divide nex 
parser.add_argument('--dataset', type=str, default='synthetic', choices=['synthetic', 'flickr', ''])
parser.add_argument('--l2', type=float, default=.0005)
parser.add_argument('--model', type=str, default='spen', choices=['spen', 'mlp', ''])
parser.add_argument('--nex', type=int, default=1500) # num_examples
#parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'momentum', 'sgd'])
parser.add_argument('--num_epochs', type=int, default=30)
#parser.add_argument('--num_train_steps', type=int, default=1000)
parser.add_argument('--num_inner_steps', type=int, default=50)
# TODO:
parser.add_argument('--pretrain', type=str, default='none', choices=['none', 'features', 'features_global', 'features_global_joint'],
        help="features: pretrain the feature network to predict y;\n
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

# TODO:
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
# TODO: how to relax the constraint on the Ys?? (make it more dependent on X)
# TODO: splits
if dataset == 'synthetic':
    X = np.random.randn(nex, input_size).astype('float32')
    A = np.random.randn(input_size, 16).astype('float32')
    Z = np.dot(X, A) # (N, 16)
    Z4x4 = Z.reshape((-1, 4, 4))
    Y_true = onehot(np.argmax(Z4x4, axis=-1)).reshape((-1, 16)).astype('float32')
    # VALID SET
    Xv = np.random.randn(nex, input_size).astype('float32')
    Av = np.random.randn(input_size, 16).astype('float32')
    Zv = np.dot(X, A) # (N, 16)
    Z4x4v = Z.reshape((-1, 4, 4))
    Y_truev = onehot(np.argmax(Z4x4, axis=-1)).reshape((-1, 16)).astype('float32')




# ---------------------------------------------------------------
print "SET-UP TRAINING"
# See sections 3 and 7.3


# TODO: use shared variables and indexing (for GPU)
x = T.matrix('float32')
x.tag.test_value = X[:batch_size]
# following Belanger et al. (2017), we optimize the logit instead of doing mirror descent
# mirror descent: see SLJ's lecture notes (19) (but how do we keep y < 1???)
#    y_update = y * T.exp(-.1 * T.grad(energy_y)) (sign??)
y_logit = theano.shared(np.zeros((batch_size, num_labels)).astype('float32'), name='y_logit')
y_bar = T.nnet.sigmoid(y_logit)
y_true = T.matrix('float32')
y_true.tag.test_value = Y_true[:batch_size]

params = []


#---------------
# Compute ENERGY (= -score)

# feature net
feature_Ws = [parameter((input_size, num_labels), 'feature_W')]
params += feature_Ws
feature_bs = [parameter((num_labels,), 'feature_b', 'bias')]
params += feature_bs
features = T.dot(x, feature_Ws[0]) + feature_bs[0]
pretraining_loss = categorical_crossentropy(y_true, T.nnet.sigmoid(features)).mean()
# zero-one loss
pretraining_error = T.neq(y_true, T.cast(T.nnet.sigmoid(features)>.5, 'float32')).mean()
opt = SGD(lr=.01, momentum=.9)
pretraining_step = theano.function( inputs=[x, y_true], 
                                      outputs=[pretraining_loss, pretraining_error],
                                      updates=opt.get_updates(feature_Ws + feature_bs, [pretraining_loss, pretraining_error], (pretraining_loss)) )
# this is used to initialize y_logits
get_features = theano.function( inputs=[x],
                                outputs=features)

# local energy
#local_b = parameter((num_labels, 1), 'local_b')
local_b = parameter((num_labels, ), 'local_b')
params += [local_b]
local_energy = lambda y : T.sum(y * T.dot(features, local_b).reshape((-1,1)), axis=-1)

# global energy
global_Ws = []
global_bs = []
global_Ws += [parameter((num_labels, 4), 'global_W0')]
global_bs += [parameter((4,), 'global_b0', 'bias')]
#global_Ws += [parameter((4, 1), 'global_W1')]
global_Ws += [parameter((4,), 'global_W1')]
global_bs += [parameter((1,), 'global_b1', 'scalar')]
params += global_Ws
params += global_bs
def global_energy(y):
    rval = hard_tanh(T.dot(y, global_Ws[0]) + global_bs[0])
    return T.dot(rval, global_Ws[1]) + global_bs[1]

energy = lambda y : local_energy(y) + global_energy(y)
energy_y_bar = energy(y_bar)
energy_y_true = energy(y_true)

#----------------------------
# Losses and THEANO functions

y_pred = T.cast(y_bar > .5, 'float32')
zero_one_loss = T.neq(y_pred, y_true).mean(-1)
# we approximate the 0-1 loss with log-loss (Section 4, 2nd column)
surrogate_loss = categorical_crossentropy(y_true, y_bar)

# TODO: F1

# loss-augmented inference
# TODO: this can be made more efficient by pre-computing the features
y_opt = SGD(lr=.1, momentum=.95)
y_step_outputs = [y_bar, y_pred] # TODO: I could monitor everything that changes here... 
y_step = theano.function([x, y_true], y_step_outputs, updates=y_opt.get_updates([y_logit], [], (surrogate_loss + energy_y_bar).mean()))
y_stepv = theano.function([x], y_step_outputs, updates=y_opt.get_updates([y_logit], [], (energy_y_bar).mean()))


# SSVM loss; energy_y is **approximately** minimized by iterative inference (the y_step function above)
train_loss = surrogate_loss + energy_y_true - energy_y_bar
train_loss = train_loss * (train_loss > 0)
train_opt = SGD(lr=.01, momentum=.9)
train_updates = train_opt.get_updates(params, [], train_loss.mean())
train_step_outputs = [train_loss, surrogate_loss, energy_y_true, energy_y_bar, y_true, y_pred, zero_one_loss]
train_step = theano.function([x, y_true], train_step_outputs, updates=train_updates)

# evaluation
evaluation = theano.function([x, y_true], train_step_outputs)

#TODO mlp_baseline = 64, 64, 16, 16


# ---------------------------------------------------------------
print "TRAIN (and SAVE)"

"""
Pretraining:
    train feature net to predict y_true directly
Training (everything is done using mini-batches in Belanger's paper!):
    get features(x)
    initialize y (as output of MLP, or .5)
    optimize y until stopping criterion
    compute and descend loss(x, y_true) using y
"""

num_batches = nex / batch_size
num_train_steps = num_epochs * num_batches

# ----------
# MONITORING (TODO: f-measure)
# TODO: names!
y_step_monitored = np.empty((num_train_steps, batch_size, num_inner_steps, len(y_step_outputs), 16)) # TODO 16
train_step_monitored = np.empty((num_train_steps, batch_size, len(train_step_outputs)))

# ----------
# PRETRAINING
if pretrain: # TODO: how long to do it, etc.
    for epoch in range(num_epochs):
        shuffles = np.random.permutation(nex)
        X = X[shuffles]
        Y_true = Y_true[shuffles]
        for batch in range(num_batches): 
            XX = X[batch*batch_size: (batch+1)*batch_size]
            YY = Y_true[batch*batch_size: (batch+1)*batch_size]
            print pretraining_step(XX, YY)
    
    #assert False # TODO


# training:
step = 0
for epoch in range(num_epochs):
    shuffles = np.random.permutation(nex)
    X = X[shuffles]
    Y_true = Y_true[shuffles]
    for batch in range(num_batches): 
        XX = X[batch*batch_size: (batch+1)*batch_size]
        YY = Y_true[batch*batch_size: (batch+1)*batch_size]
        # initialize y
        #y_logit.set_value((.0 * np.ones((batch_size, num_labels))).astype('float32'))
        y_logit.set_value(get_features(XX))
        # optimize y
        for inner_step in range(num_inner_steps):
            monitored = y_step(XX, YY)
            y_step_monitored[step, :, inner_step, :] = np.array(monitored).transpose(1,0,2)
        monitored = train_step(XX, YY)
        if 0: # train set is perfect 
            print "01", monitored[-1].mean()
        #train_step_outputs = [train_loss, surrogate_loss, energy_y_true, energy_y_bar, zero_one_loss]
        # FIXME: shapes
        #train_step_monitored[step, :, :] = np.array(monitored)
        step += 1

        #----------------- 
        # validation
        if batch == num_batches - 1:
            print "01", monitored[-1].mean()
            XX = Xv[batch*batch_size: (batch+1)*batch_size]
            YY = Y_truev[batch*batch_size: (batch+1)*batch_size]
            # initialize y
            y_logit.set_value((.0 * np.ones((batch_size, num_labels))).astype('float32'))
            # optimize y (at test time, we aren't allowed to see the ground truth; we just optimize the energy_
            for inner_step in range(num_inner_steps):
                monitored = y_stepv(XX)
                #print monitored
                y_step_monitored[step, :, inner_step, :] = np.array(monitored).transpose(1,0,2)
            monitored = evaluation(XX, YY)


assert False

# ---------------------------------------------------------------
# ANALYZE (e.g. PLOTS)

# As a sanity check, we expect to see the following:
#   Energy goes down during inner_step
print energy.mean(-1)[0]
print energy.mean(-1)[-1]
print energy.mean(-1).mean(0)
#   Energy of y_true goes down over time
#   Energy of y_bar goes UP if the prediction is wrong
#   When the loss is low (e.g. zero), we should be making good predictions (unless the inner_opt is failing)
#       This was not the case ATM
print 

