#!/usr/bin/env python

"""
TODO:
    baseline MLP
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

"""

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
parser.add_argument('--dataset', type=str, default='synthetic', choices=['synthetic', 'flickr', 'test'])
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--lr', type=float, default=.0001)
parser.add_argument('--lr_inner', type=float, default=1.)
parser.add_argument('--lr_local', type=float, default=.01)
parser.add_argument('--l2', type=float, default=1e-4)
parser.add_argument('--model', type=str, default='spen', choices=['spen', 'mlp', '']) # TODO: mlp
parser.add_argument('--nex', type=int, default=1500) # num_examples
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--num_experiments', type=int, default=10)
parser.add_argument('--num_inner_steps', type=int, default=100)
parser.add_argument('--pretrain', type=str, default='local_and_global', choices=['none', 'local', 'local_and_global'])
parser.add_argument('--y_init', type=str, default='features', choices=['0s', 'features'])
#
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--save_dir', type=str, default="./")
parser.add_argument('--seed', type=int, default=None)
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



pretrains = ['none', 'local', 'local_and_global']

if 1:

    all_monitored = np.empty((len(pretrains), num_experiments, num_epochs))
    all_monitoredv = np.empty((len(pretrains), num_experiments, num_epochs))
    all_monitoredt = np.empty((len(pretrains), num_experiments, num_epochs))

    for pt, pretrain in enumerate(pretrains):
        for seed in range(num_experiments):

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


            def get_l2(theano_vars):
                l2 = 0.
                for var in theano_vars:
                    l2 += ((var.flatten() ** 2).sum())**.5
                return l2


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
                Zv = np.dot(Xv, A) # (N, 16)
                Z4x4v = Zv.reshape((-1, 4, 4))
                Y_truev = onehot(np.argmax(Z4x4v, axis=-1)).reshape((-1, 16)).astype('float32')
                # TEST SET
                Xt = np.random.randn(nex, input_size).astype('float32')
                Zt = np.dot(Xt, A) # (N, 16)
                Z4x4t = Zt.reshape((-1, 4, 4))
                Y_truet = onehot(np.argmax(Z4x4t, axis=-1)).reshape((-1, 16)).astype('float32')

            elif dataset == 'test': # a bit easier... just the argmax
                X = np.random.randn(nex, input_size).astype('float32')
                A = np.random.randn(input_size, num_labels).astype('float32')
                Y = np.dot(X, A) # (N, 16)
                Y_true = onehot(np.argmax(Y, axis=-1)).reshape((-1, 16)).astype('float32')
                # VALID SET
                Xv = np.random.randn(nex, input_size).astype('float32')
                Yv = np.dot(Xv, A) # (N, 16)
                Y_truev = onehot(np.argmax(Yv, axis=-1)).reshape((-1, 16)).astype('float32')



            def shuffle_data():
                global X, Xv, Y_true, Y_truev
                shuffles = np.random.permutation(nex)
                X = X[shuffles]
                Y_true = Y_true[shuffles]
                shuffles = np.random.permutation(nex)
                Xv = Xv[shuffles]
                Y_truev = Y_truev[shuffles]

            # ---------------------------------------------------------------
            print "SET-UP TRAINING"
            # See sections 3, 7.3, A.4, A.5


            x = T.matrix('float32')
            x.tag.test_value = X[:batch_size]
            # following Belanger et al. (2017), we optimize the logit instead of doing mirror descent
            y_logit = theano.shared(np.zeros((batch_size, num_labels)).astype('float32'), name='y_logit')
            #y_logit_full_batch = theano.shared(np.zeros((batch_size, num_labels)).astype('float32'), name='y_logit')
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
                                                  updates=SGD(lr=lr_local, momentum=.9).get_updates(local_params, [], local_pretraining_loss) )
            # evaluation (no updates, i.e. no learning)
            local_pretraining_stepv = theano.function( inputs=[x, y_true], 
                                                  outputs=[local_pretraining_loss, local_pretraining_error, local_pretraining_preds])
            # this is used to initialize y_logits (section?)
            get_features = theano.function( inputs=[x], outputs=features)

            # TODO: double check this!!!
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
            #global_bs += [parameter((1,), 'global_b1', 'scalar')]
            global_params = global_Ws + global_bs
            def global_energy(y):
                rval = hard_tanh(T.dot(y, global_Ws[0]) + global_bs[0])
                return T.dot(rval, global_Ws[1])# + global_bs[1]

            params = local_params + global_params

            energy = lambda y : local_energy(y) + global_energy(y)
            energy_y_bar = energy(y_bar)
            energy_y_true = energy(y_true)

            energy_fn = theano.function([x, y_true], [local_energy(y_true), global_energy(y_true)])


            #---------------
            print "DEFINE SSVM TRAINING"

            y_pred = T.cast(y_bar > .5, 'float32')
            zero_one_loss = T.neq(y_pred, y_true).mean(-1)
            # we approximate the 0-1 loss with log-loss (Section 4, 2nd column)
            surrogate_loss = binary_crossentropy(y_true, y_bar)


            # loss-augmented inference
            y_opt = SGD(lr=lr_inner, momentum=.9)
            y_step_outputs = [y_true, y_bar, y_pred]
            y_step = theano.function([x, y_true], y_step_outputs, updates=y_opt.get_updates([y_logit], [], (-surrogate_loss + energy_y_bar).mean()))
            # at test time, we don't get to see the ground truth, we just find the y which minimizes the energy
            y_stepv = theano.function([x, y_true], y_step_outputs, updates=y_opt.get_updates([y_logit], [], (energy_y_bar).mean()))

            # FIXME: train_loss = nan?

            # SSVM loss; energy_y is **approximately** minimized by iterative inference (the y_step function above)
            train_loss = surrogate_loss + energy_y_true - energy_y_bar
            train_loss = train_loss * (train_loss > 0)
            #global_train_loss = train_loss + l2 * get_l2(global_params)
            train_loss = train_loss * (train_loss > 0) + get_l2(params)
            train_updates = SGD(lr=lr, momentum=.9).get_updates(params, [], train_loss.mean())
            train_step_outputs = [train_loss, surrogate_loss, energy_y_true, energy_y_bar, y_true, y_pred, zero_one_loss]
            outs_str = "[train_loss, surrogate_loss, energy_y_true, energy_y_bar, y_true, y_pred, zero_one_loss]"
            train_step = theano.function([x, y_true], train_step_outputs, updates=train_updates)

            # for the global training, we only update the global params
            global_train_updates = SGD(lr=.01, momentum=.9).get_updates(global_params, [], train_loss.mean())
            global_train_step = theano.function([x, y_true], train_step_outputs, updates=train_updates)

            # evaluation (no updates, i.e. no learning)
            train_stepv = theano.function([x, y_true], train_step_outputs)

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
            monitoredt = np.empty((num_epochs, 3))

            # -------------------------
            if 'local' in pretrain:
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


                
            if 'global' in pretrain:
                print "\nPRETRAINING global energy\n"
                for epoch in range(num_epochs):
                    # TRAINING STEPS
                    shuffle_data()
                    for batch in range(num_batches): 
                        XX = X[batch*batch_size: (batch+1)*batch_size]
                        YY = Y_true[batch*batch_size: (batch+1)*batch_size]
                        # initialize y
                        if y_init == '0s':
                            y_logit.set_value((.0 * np.ones((batch_size, num_labels))).astype('float32'))
                        else:
                            y_logit.set_value(get_features(XX))
                        # optimize y
                        for inner_step in range(num_inner_steps):
                            y_step(XX, YY)
                        global_train_step(XX, YY)

                    # MONITORING
                    # train
                    # initialize y
                    if y_init == '0s':
                        y_logit.set_value((.0 * np.ones((batch_size, num_labels))).astype('float32'))
                    else:
                        y_logit.set_value(get_features(XX))
                    # optimize y
                    for inner_step in range(num_inner_steps):
                        y_stepv(XX, YY)
                    # outs = [train_loss, surrogate_loss, energy_y_true, energy_y_bar, y_true, y_pred, zero_one_loss]
                    outs = train_stepv(XX, YY)
                    global_monitored[epoch] = outs[0].mean(), outs[-1].mean(), np.mean([f1_score(outs[-3][nn], outs[-2][nn], average='macro') for nn in range(batch_size)])

                    # valid
                    XX = Xv[batch*batch_size: (batch+1)*batch_size]
                    YY = Y_truev[batch*batch_size: (batch+1)*batch_size]
                    # initialize y
                    if y_init == '0s':
                        y_logit.set_value((.0 * np.ones((batch_size, num_labels))).astype('float32'))
                    else:
                        y_logit.set_value(get_features(XX))
                    # optimize y
                    for inner_step in range(num_inner_steps):
                        y_stepv(XX, YY)
                    # outs = [train_loss, surrogate_loss, energy_y_true, energy_y_bar, y_true, y_pred, zero_one_loss]
                    outs = train_stepv(XX, YY)
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

                        if debug:
                            en = energy_fn(XX,YY)
                            print ""
                            print en
                            print global_params
                            import ipdb; ipdb.set_trace()

                        # initialize y
                        if y_init == '0s':
                            y_logit.set_value((.0 * np.ones((batch_size, num_labels))).astype('float32'))
                        else:
                            y_logit.set_value(get_features(XX))
                        # optimize y
                        for inner_step in range(num_inner_steps):
                            yt, yb, yp = y_step(XX, YY)
                            if debug:
                                print ""
                                print yt[0]
                                print yb[0]
                                print yp[0]

                        train_step(XX, YY)

                    # MONITORING
                    # train
                    # initialize y
                    if y_init == '0s':
                        y_logit.set_value((.0 * np.ones((batch_size, num_labels))).astype('float32'))
                    else:
                        y_logit.set_value(get_features(XX))
                    # optimize y
                    for inner_step in range(num_inner_steps):
                        y_stepv(XX, YY)
                    # outs = [train_loss, surrogate_loss, energy_y_true, energy_y_bar, y_true, y_pred, zero_one_loss]
                    outs = train_stepv(XX, YY)
                    #print outs
                    monitored[epoch] = outs[0].mean(), outs[-1].mean(), np.mean([f1_score(outs[-3][nn], outs[-2][nn], average='macro') for nn in range(batch_size)])

                    # valid
                    XX = Xv[batch*batch_size: (batch+1)*batch_size]
                    YY = Y_truev[batch*batch_size: (batch+1)*batch_size]
                    # initialize y
                    if y_init == '0s':
                        y_logit.set_value((.0 * np.ones((batch_size, num_labels))).astype('float32'))
                    else:
                        y_logit.set_value(get_features(XX))
                    # optimize y
                    for inner_step in range(num_inner_steps):
                        y_stepv(XX, YY)
                    # outs = [train_loss, surrogate_loss, energy_y_true, energy_y_bar, y_true, y_pred, zero_one_loss]
                    outs = train_stepv(XX, YY)
                    monitoredv[epoch] = outs[0].mean(), outs[-1].mean(), np.mean([f1_score(outs[-3][nn], outs[-2][nn], average='macro') for nn in range(batch_size)])

                    # test
                    XX = Xt[batch*batch_size: (batch+1)*batch_size]
                    YY = Y_truet[batch*batch_size: (batch+1)*batch_size]
                    # initialize y
                    if y_init == '0s':
                        y_logit.set_value((.0 * np.ones((batch_size, num_labels))).astype('float32'))
                    else:
                        y_logit.set_value(get_features(XX))
                    # optimize y
                    for inner_step in range(num_inner_steps):
                        y_stepv(XX, YY)
                    # outs = [train_loss, surrogate_loss, energy_y_true, energy_y_bar, y_true, y_pred, zero_one_loss]
                    outs = train_stepv(XX, YY)
                    monitoredt[epoch] = outs[0].mean(), outs[-1].mean(), np.mean([f1_score(outs[-3][nn], outs[-2][nn], average='macro') for nn in range(batch_size)])

                    print "(train)  loss, error, F1:  ", monitored[epoch],'           (valid):  ', monitoredv[epoch]







            all_monitored[pt, seed] = monitored[:,-1]
            all_monitoredv[pt, seed] = monitoredv[:,-1]
            all_monitoredt[pt, seed] = monitoredt[:,-1]


# ---------------------------------------------------------------
# ANALYZE (e.g. PLOTS)

if 1:
    best_epochs = np.argmax(all_monitoredv, axis=-1)

    test_perfs = np.empty((len(pretrains), num_experiments))
    for pt in range(len(pretrains)):
        for nexp in range(num_experiments):
            test_perfs[pt, nexp] = all_monitoredt[pt, nexp, best_epochs[pt, nexp]]
        print test_perfs[pt].mean()
        print test_perfs[pt].std() / (num_experiments**.5)


    #from utils import err_plot




np.save('test_perfs', test_perfs)


from pylab import *

N = len(pretrains)
men_means = [test_perfs[pt].mean() for pt in range(len(pretrains))]
men_std = [test_perfs[pt].std() / (num_experiments**.5) for pt in range(len(pretrains))]

ind = np.arange(N)  # the x locations for the groups
width = 0.5       # the width of the bars

fig, ax = subplots()
rects1 = ax.bar(ind, men_means, width, color='r', yerr=men_std)

# add some text for labels, title and axes ticks
ax.set_ylabel('F1 measure')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('no pretraining', 'local pretraining', 'local and global pretraining'))

savefig('SLJ_')

show()





