from __future__ import print_function
import argparse
import logging
import numpy as np
from time import time
import sys
import chatbotscorer.utils as U
import chatbotscorer.helper as helper
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
parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True, help="The path to the output directory")
parser.add_argument("-t", "--model-type", dest="model_type", type=str, metavar='<str>', default='cnn', help="Model type (rnn|cnn|rcnn|crnn|cwrnn|cnn+cnn|rnn+cnn) (default=rnn)")
parser.add_argument("-p", "--pooling-type", dest="pooling_type", type=str, metavar='<str>', default=None, help="Pooling type (meanot|attsum|attmean) (default=meanot)")
parser.add_argument("-u", "--rec-unit", dest="recurrent_unit", type=str, metavar='<str>', default='lstm', help="Recurrent unit type (lstm|gru|simple) (default=lstm)")
parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='rmsprop', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=rmsprop)")
parser.add_argument("-lr", "--learning-rate", dest="rmsprop_learning_rate", type=float, metavar='<float>', default=0.001, help="Learning rate in rmsprop (default=0.001)")
parser.add_argument("-rho", "--rho", dest="rmsprop_rho", type=float, metavar='<float>', default=0.9, help="rho in rmsprop (default=0.9)")
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=50, help="Embeddings dimension (default=50)")
parser.add_argument("-cl","--cnn-layer", dest="cnn_layer", type=int, metavar='<int>', default=1, help="Number of CNN layer (default=1)")
parser.add_argument("-c", "--cnndim", dest="cnn_dim", type=int, metavar='<int>', default=0, help="CNN output dimension. '0' means no CNN layer (default=0)")
parser.add_argument("-w", "--cnnwin", dest="cnn_window_size", type=int, metavar='<int>', default=2, help="CNN window size. (default=2)")
parser.add_argument("-rl","--rnn-layer", dest="rnn_layer", type=int, metavar='<int>', default=1, help="Number of RNN layer (default=1)")
parser.add_argument("-r", "--rnndim", dest="rnn_dim", type=int, metavar='<int>', default=300, help="RNN dimension. '0' means no RNN layer (default=300)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=32, help="Batch size for training (default=32)")
parser.add_argument("-be", "--batch-size-eval", dest="batch_size_eval", type=int, metavar='<int>', default=256, help="Batch size for evaluation (default=256)")
parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=4000, help="Vocab size (default=4000)")
parser.add_argument("-do", "--dropout", dest="dropout_rate", type=float, metavar='<float>', default=0.5, help="The dropout rate in the model (default=0.5)")
parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>', help="The path to the word embeddings file (Word2Vec format)")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=50, help="Number of epochs (default=50)")
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1337, help="Random seed (default=1337)")

args = parser.parse_args()

out_dir = args.out_dir_path

U.mkdir_p(out_dir + '/data')
U.mkdir_p(out_dir + '/preds')
U.mkdir_p(out_dir + '/models')
U.set_logger(out_dir)
U.print_args(args)

####################################################################################
## Argument Validation
#

valid_model_type = {
    'cnn',
}
model_type_multiclass = {'cls'}
model_type_binary = {'rnn', 'rcnn', 'crnn', 'cwrnn', 'cwbrnn', 'cnn', 'rnn+cnn', 'cnn+cnn', 'brnn', 'bclslab'}

assert args.model_type in valid_model_type
assert args.algorithm in {'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'}
assert args.recurrent_unit in {'lstm', 'gru', 'simple'}
assert args.pooling_type in { None, 'meanot', 'attsum', 'attmean'}

import chatbotscorer.dataset_reader as dataset_reader
from chatbotscorer.evaluator import Evaluator

#######################################################################################
## Prepare data
#

((train_x, train_y, train_lab_x, train_filename_y),
 (dev_x, dev_y, dev_lab_x, dev_filename_y),
 vocab, vocab_size, overal_maxlen) = dataset_reader.load_dataset(args)

# Initialize Embedding reader
from chatbotscorer.models import load_embedding_reader
emb_reader = load_embedding_reader(args)

############################################################################################
## Initialize Evaluator
## WARNING! Must be initialize Evaluator before padding to do dynamic padding
#

evl = Evaluator(
    logger, out_dir,
    (train_x, train_lab_x, train_y, train_filename_y),
    (dev_x, dev_lab_x, dev_y, dev_filename_y),
    (test_x, test_lab_x, test_y, test_filename_y),
    args.model_type, batch_size_eval=args.batch_size_eval)

############################################################################################
## Padding to dataset for statistics
#

if args.seed > 0:
    logger.info('Setting np.random.seed(%d) before importing keras' % args.seed)
    np.random.seed(args.seed)

import keras.backend as K
from keras.preprocessing import sequence

# Pad sequences for mini-batch processing
padded_train_x = sequence.pad_sequences(train_x)
padded_dev_x = sequence.pad_sequences(dev_x)
padded_test_x = sequence.pad_sequences(test_x)

############################################################################################
## Some statistics
#

bincount = np.bincount(train_y)
most_frequent_class = bincount.argmax()
np.savetxt(out_dir + '/preds/bincount.txt', bincount, fmt='%i')

np_train_y = np.array(train_y, dtype=K.floatx())
np_dev_y = np.array(dev_y, dtype=K.floatx())
np_test_y = np.array(test_y, dtype=K.floatx())

train_mean = np_train_y.mean()
train_std = np_train_y.std()
dev_mean = np_dev_y.mean()
dev_std = np_dev_y.std()
test_mean = np_test_y.mean()
test_std = np_test_y.std()

logger.info('Statistics:')

logger.info('  train_x shape: ' + str(np.array(padded_train_x).shape))
logger.info('  dev_x shape:   ' + str(np.array(padded_dev_x).shape))
logger.info('  test_x shape:  ' + str(np.array(padded_test_x).shape))

logger.info('  train_y shape: ' + str(np_train_y.shape))
logger.info('  dev_y shape:   ' + str(np_dev_y.shape))
logger.info('  test_y shape:  ' + str(np_test_y.shape))

logger.info('  train_y mean: %.3f, stdev: %.3f, MFC: %i' % (train_mean, train_std, most_frequent_class))

############################################################################################
## Compute class weight (where data is usually imbalanced)
# Always imbalanced in medical text data

# class_weight = helper.compute_class_weight(np.array(train_y, dtype=K.floatx()))

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

from chatbotscorer.models import create_model
model = create_model(args, overal_maxlen, vocab, emb_reader)
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

total_train_time = 0
total_eval_time = 0

for ii in range(args.epochs):
    t0 = time()

    # Train in chunks of batch size and dynamically padded
    train_history = model.fit(train_x, train_y, batch_size=args.batch_size, nb_epoch=1, verbose=0)
        
    tr_time = time() - t0
    total_train_time += tr_time

    # Evaluate
    t0 = time()
    evl.evaluate(model, ii)
    evl_time = time() - t0
    total_eval_time += evl_time

    logger.info('Epoch %d, train: %is (%.1fm), evaluation: %is (%.1fm)' % (ii, tr_time, tr_time/60.0, evl_time, evl_time/60.0))
    logger.info('[Train] loss: %.4f' % (train_loss))

    # Print and send email Epoch LSTM
    content = evl.print_info()

###############################################################################################################################
## Summary of the results
#

total_time = total_train_time + total_eval_time

total_train_time_hours = total_train_time/3600
total_eval_time_hours = total_eval_time/3600
total_time_hours = total_time/3600

logger.info('Training:   %i seconds in total (%.1f hours)' % (total_train_time, total_train_time_hours))
logger.info('Evaluation: %i seconds in total (%.1f hours)' % (total_eval_time, total_eval_time_hours))
logger.info('Total time: %i seconds in total (%.1f hours)' % (total_time, total_time_hours))
logger.info('---------------------------------------------------------------------------------------')

logger.info('Missed @ Epoch %i:' % evl.best_test_missed_epoch)
logger.info('  [TEST] F1: %.3f' % evl.best_test_missed)
logger.info('Best @ Epoch %i:' % evl.best_dev_epoch)
logger.info('  [DEV]  F1: %.3f, Recall: %.3f, Precision: %.3f, Spec: %.5f' % (evl.best_dev[0], evl.best_dev[1], evl.best_dev[2], evl.best_dev[3]))
logger.info('  [TEST] F1: %.3f, Recall: %.3f, Precision: %.3f, Spec: %.5f' % (evl.best_test[0], evl.best_test[1], evl.best_test[2], evl.best_test[3]))

content = ('\r\n\r\nMissed @ Epoch %i: \r\n\r\n' % evl.best_test_missed_epoch)
content = content + ('  [TEST] F1: %.3f \r\n\r\n' % evl.best_test_missed)
content = content + ('Best @ Epoch %i: \r\n\r\n' % evl.best_dev_epoch)
content = content + ('  [DEV]  F1: %.3f, Recall: %.3f, Precision: %.3f, Spec: %.5f \r\n\r\n' % (evl.best_dev[0], evl.best_dev[1], evl.best_dev[2], evl.best_dev[3]))
content = content + ('  [TEST] F1: %.3f, Recall: %.3f, Precision: %.3f, Spec: %.5f \r\n\r\n' % (evl.best_test[0], evl.best_test[1], evl.best_test[2], evl.best_test[3]))
