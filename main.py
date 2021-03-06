from __future__ import print_function
import argparse
import logging
import numpy as np
from time import time
import sys
import chatbotscorer.utils as U
import chatbotscorer.helper as helper
import chatbotscorer.svmrf as svmrf
import pickle as pk
import os.path

logger = logging.getLogger(__name__)

###############################################################################################################################
## Parse arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("-tr", "--train", dest="train_path", type=str, metavar='<str>', default=None, help="The path to the training set")
parser.add_argument("-tu", "--tune", dest="dev_path", type=str, metavar='<str>', default=None, help="The path to the development set")
parser.add_argument("-ts", "--test", dest="test_path", type=str, metavar='<str>', default=None, help="The path to the test set")
parser.add_argument("-lt", "--label-type", dest="label_type", type=str, metavar='<str>', default='max', help="Label type, how to handle multiple different labels (max|min|mean) (default=max)")
parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True, help="The path to the output directory")
parser.add_argument("-t", "--model-type", dest="model_type", type=str, metavar='<str>', default='cnn', help="Model type (rnn|cnn|rcnn|crnn|cwrnn|cnn+cnn|rnn+cnn) (default=rnn)")
parser.add_argument("-p", "--pooling-type", dest="pooling_type", type=str, metavar='<str>', default=None, help="Pooling type (meanot|attsum|attmean) (default=meanot)")
parser.add_argument("-u", "--rec-unit", dest="recurrent_unit", type=str, metavar='<str>', default='lstm', help="Recurrent unit type (lstm|gru|simple) (default=lstm)")
parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='rmsprop', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=rmsprop)")
parser.add_argument("-lr", "--learning-rate", dest="rmsprop_learning_rate", type=float, metavar='<float>', default=0.001, help="Learning rate in rmsprop (default=0.001)")
parser.add_argument("-rho", "--rho", dest="rmsprop_rho", type=float, metavar='<float>', default=0.9, help="rho in rmsprop (default=0.9)")
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=50, help="Embeddings dimension (default=50)")
parser.add_argument("-cl","--cnn-layer", dest="cnn_layer", type=int, metavar='<int>', default=1, help="Number of CNN layer (default=1)")
parser.add_argument("-c", "--cnndim", dest="cnn_dim", type=int, metavar='<int>', default=300, help="CNN output dimension. '0' means no CNN layer (default=0)")
parser.add_argument("-w", "--cnnwin", dest="cnn_window_size", type=int, metavar='<int>', default=2, help="CNN window size. (default=2)")
parser.add_argument("-rl","--rnn-layer", dest="rnn_layer", type=int, metavar='<int>', default=1, help="Number of RNN layer (default=1)")
parser.add_argument("-r", "--rnndim", dest="rnn_dim", type=int, metavar='<int>', default=300, help="RNN dimension. '0' means no RNN layer (default=300)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=32, help="Batch size for training (default=32)")
parser.add_argument("-be", "--batch-size-eval", dest="batch_size_eval", type=int, metavar='<int>', default=256, help="Batch size for evaluation (default=256)")
parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=4000, help="Vocab size (default=4000)")
parser.add_argument("-cw", "--class-weight", dest="is_class_weight", action='store_true', help="Flag to use class weight (default=False)")
parser.add_argument("-do", "--dropout", dest="dropout_rate", type=float, metavar='<float>', default=0.5, help="The dropout rate in the model (default=0.5)")
parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>', help="The path to the word embeddings file (Word2Vec format)")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=50, help="Number of epochs (default=50)")
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1337, help="Random seed (default=1337)")
parser.add_argument("-dd", "--dump-data", dest="is_dump_data", action='store_true', help="Flag to use to dump train, valid, and test data for all folds")
parser.add_argument("-sd", "--show-distribution", dest="is_show_distribution", action='store_true', help="Flag to show the distribution (count) of the ground truth in the dataset")
parser.add_argument("-sw", "--show-weights", dest="is_show_weights", action='store_true', help="Flag to show Random Forest top 10 weights (most important word)")


args = parser.parse_args()

out_dir = args.out_dir_path

U.mkdir_p(out_dir + '/data')
if (args.is_dump_data):
    U.mkdir_p(out_dir + '/data/dump')
U.mkdir_p(out_dir + '/preds')
U.mkdir_p(out_dir + '/models')
U.set_logger(out_dir)
U.print_args(args)

####################################################################################
## Argument Validation
#

valid_model_type = {
    'svm',
    'rf',
    'cnn',
    'rnn'
}

assert args.model_type in valid_model_type
assert args.algorithm in {'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'}
assert args.recurrent_unit in {'lstm', 'gru', 'simple'}
assert args.pooling_type in { None, 'meanot', 'attsum', 'attmean'}
assert args.label_type in { 'max', 'min', 'mean'}

import chatbotscorer.reader as dataset_reader
from chatbotscorer.evaluator import Evaluator

#######################################################################################
## Prepare data
#

(global_train_x, global_train_y,
 global_dev_x, global_dev_y,
 global_test_x, global_test_y,
 global_original_test_x,
 global_vocab, global_maxlen) = dataset_reader.load_dataset(args)

###########################################################################################
## 10 FOLD CROSS VALIDATION
#
total_f1 = 0
total_acc = 0
total_maj_f1 = 0
total_maj_acc = 0

total_correlation = 0
total_p_value = 0
total_test_pred = np.array([])
total_test_y = np.array([])

total_train_time = 0
total_eval_time = 0

##################################
## Show Ground Truth Distribution/Count
#
if (args.is_show_distribution):
    train_y = global_train_y[0]
    dev_y = global_dev_y[0]
    test_y = global_test_y[0]
    y = np.concatenate((train_y,dev_y,test_y))

    freq_table = dict()
    for curr_scores in y:
        if curr_scores in freq_table.keys():
            freq_table[curr_scores] = freq_table[curr_scores]  + 1
        else:
            freq_table[curr_scores] = 1
    logger.info("=========== Distribution of dataset ============")
    for key, value in sorted(freq_table.items()):
        logger.info("{} : {}".format(key, value))
    logger.info("Total dataset size: " + str(len(y)))
#########################################

for fold in range(10):
    logger.info("========================== FOLD %i ===============================" % fold)
    
    train_x = global_train_x[fold]
    train_y = global_train_y[fold]
    dev_x = global_dev_x[fold]
    dev_y = global_dev_y[fold]
    test_x = global_test_x[fold]
    test_y = global_test_y[fold]
    vocab = global_vocab[fold]
    maxlen = global_vocab[fold]

    original_test_x = global_original_test_x[fold] # To see real-life cases
    
    if args.model_type == 'svm' or args.model_type == 'rf':
        test1, test2 = svmrf.run_svmrf(args, fold, train_x, train_y, test_x, test_y, vocab=vocab)
        if args.label_type == 'mean':
            total_correlation += test1
            total_p_value += test2
        else:
            total_f1 += test1
            total_acc += test2
        continue

    ############################################################################################
    ## Padding to dataset for statistics
    #

    if args.seed > 0:
        logger.info('Setting np.random.seed(%d) before importing keras' % args.seed)
        np.random.seed(args.seed)

    import keras.backend as K
    from keras.preprocessing import sequence

    # Pad sequences for mini-batch processing
    train_x[0] = sequence.pad_sequences(train_x[0])
    train_x[1] = sequence.pad_sequences(train_x[1])
    dev_x[0] = sequence.pad_sequences(dev_x[0])
    dev_x[1] = sequence.pad_sequences(dev_x[1])
    test_x[0] = sequence.pad_sequences(test_x[0])
    test_x[1] = sequence.pad_sequences(test_x[1])

    ############################################################################################
    ## Some statistics
    #

    bincount = np.bincount(train_y)
    most_frequent_class = bincount.argmax()
    np.savetxt(out_dir + '/preds/bincount.txt', bincount, fmt='%i')

    train_y = np.array(train_y, dtype=K.floatx())
    dev_y = np.array(dev_y, dtype=K.floatx())
    test_y = np.array(test_y, dtype=K.floatx())

    train_mean = train_y.mean()
    train_std = train_y.std()
    dev_mean = dev_y.mean()
    dev_std = dev_y.std()
    test_mean = test_y.mean()
    test_std = test_y.std()

    logger.info('Statistics:')

    logger.info('  train_x USER shape:   ' + str(np.array(train_x[0]).shape))
    logger.info('  train_x SYSTEM shape: ' + str(np.array(train_x[1]).shape))
    logger.info('  dev_x USER shape:     ' + str(np.array(dev_x[0]).shape))
    logger.info('  dev_x SYSTEM shape:   ' + str(np.array(dev_x[1]).shape))
    logger.info('  test_x USER shape:    ' + str(np.array(test_x[0]).shape))
    logger.info('  test_x SYSTEM shape:  ' + str(np.array(test_x[1]).shape))

    logger.info('  train_y shape:        ' + str(train_y.shape))
    logger.info('  dev_y shape:          ' + str(dev_y.shape))
    logger.info('  test_y shape:         ' + str(test_y.shape))

    logger.info('  train_y mean: %.3f, stdev: %.3f, MFC: %i' % (train_mean, train_std, most_frequent_class))

    ############################################################################################
    ## Initialize Evaluator
    #
    
    evl = Evaluator(
        logger, out_dir,
        (train_x, train_y),
        (dev_x, dev_y),
        (test_x, test_y),
        original_test_x,
        args.model_type,
        args.label_type,
        fold,
        batch_size_eval=args.batch_size_eval)


    ############################################################################################
    ## Compute class weight (where data is usually imbalanced)
    #

    class_weight = None
    if args.is_class_weight:
        class_weight = helper.compute_class_weight(np.array(train_y, dtype=K.floatx()))

    ###################################################################################################
    ## Optimizer algorithm
    #

    from chatbotscorer.optimizers import get_optimizer
    optimizer = get_optimizer(args)

    ####################################################################################################
    ## Building model
    #

    loss = 'binary_crossentropy'
    metric = 'accuracy'
    if args.label_type == 'mean':
        loss = 'mean_squared_error'
        metric = 'mean_absolute_error'

    from chatbotscorer.models import create_model
    model = create_model(args, maxlen, vocab)
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])
    logger.info('model compilation completed!')

    logger.info('Total number of parameters: %i' % model.count_params())
    model.summary()

    #####################################################################################################
    ## Plotting model
    #

    from keras.utils.visualize_util import plot
    plot(model, to_file = out_dir + '/models/model.png')

    ###############################################################################################################################
    ## Save model architecture
    #

    logger.info('Saving model architecture')
    with open(out_dir + '/models/model_arch.json', 'w') as arch:
        arch.write(model.to_json(indent=2))

    ###############################################################################################################################
    ## Training
    #
    logger.info('---------------------------------------------------------------------------------------')
    logger.info('Initial Evaluation:')
    evl.evaluate(model, -1)

    # Print and send email Init LSTM
    content = evl.print_info()



    for ii in range(args.epochs):
        t0 = time()

        # Train in chunks of batch size and dynamically padded
        train_history = model.fit(train_x, train_y, batch_size=args.batch_size, class_weight=class_weight, nb_epoch=1, verbose=0)
            
        tr_time = time() - t0
        total_train_time += tr_time

        # Evaluate
        t0 = time()
        evl.evaluate(model, ii)
        evl_time = time() - t0
        total_eval_time += evl_time

        logger.info('Epoch %d, train: %is (%.1fm), evaluation: %is (%.1fm)' % (ii, tr_time, tr_time/60.0, evl_time, evl_time/60.0))
        if args.label_type == 'mean':
            logger.info('[Train] loss: %.4f  MAE: %.4f' % (train_history.history['loss'][0], train_history.history['mean_absolute_error'][0]))
        else:
            logger.info('[Train] loss: %.4f  accuracy: %.4f' % (train_history.history['loss'][0], train_history.history['acc'][0]))

        evl.print_info()

    ###############################################################################################################################
    ## Summary of the results for the current fold
    #

    if args.label_type == 'mean':
        total_correlation += evl.best_test[0]
        total_p_value += evl.best_test[1]
        total_test_pred = np.append(total_test_pred, evl.test_best_pred)
        total_test_y = np.append(total_test_y, evl.test_y_org)
        
    else:
        evl.print_majority()

        total_f1 += evl.best_test[0]
        total_acc += evl.best_test[3]
        total_maj_f1 += evl.best_majority[0]
        total_maj_acc += evl.best_majority[3]
        
        logger.info('Missed @ Epoch %i:' % evl.best_test_missed_epoch)
        logger.info('  [TEST] F1: %.3f' % evl.best_test_missed)
        logger.info('Best @ Epoch %i:' % evl.best_dev_epoch)
        logger.info('  [DEV]  F1: %.3f, Recall: %.3f, Precision: %.3f, Acc: %.5f' % (evl.best_dev[0], evl.best_dev[1], evl.best_dev[2], evl.best_dev[3]))
        logger.info('  [TEST] F1: %.3f, Recall: %.3f, Precision: %.3f, Acc: %.5f' % (evl.best_test[0], evl.best_test[1], evl.best_test[2], evl.best_test[3]))


total_time = total_train_time + total_eval_time

total_train_time_hours = total_train_time/3600
total_eval_time_hours = total_eval_time/3600
total_time_hours = total_time/3600

logger.info('Training:   %i seconds in total (%.1f hours)' % (total_train_time, total_train_time_hours))
logger.info('Evaluation: %i seconds in total (%.1f hours)' % (total_eval_time, total_eval_time_hours))
logger.info('Total time: %i seconds in total (%.1f hours)' % (total_time, total_time_hours))
logger.info('---------------------------------------------------------------------------------------')

logger.info('============================================')
logger.info('Averaged Best Score across 10 folds:')
if args.label_type == 'mean':
    logger.info('  [AVG-TEST]   Correlation-coef: %.3f, p-value: %.5f ' % (total_correlation/10.0, total_p_value/10.0))
    from scipy.stats import pearsonr
    (pearson_coef, pearson_p_value) = pearsonr(total_test_pred, total_test_y)
    logger.info('  [TOTAL-TEST] Correlation-coef: %.3f, p-value: %.5f ' % (pearson_coef, pearson_p_value))
else:
    logger.info('  [MAJ]  F1: %.3f, Acc: %.5f' % (total_maj_f1/10.0, total_maj_acc/10.0))
    logger.info('  [TEST] F1: %.3f, Acc: %.5f' % (total_f1/10.0, total_acc/10.0))
    
logger.info('============================================')
