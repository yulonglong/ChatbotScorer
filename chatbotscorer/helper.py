import argparse
import logging
import numpy as np
from time import time
import sys
import utils as U
import pickle as pk

logger = logging.getLogger(__name__)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

def sort_data(x, y, filename_y, lab_x=None):
    '''Sort data based on the length of x'''

    test_lab_x = lab_x
    test_xy = zip(x, y, filename_y)
    if lab_x:
        test_xy = zip(x, y, filename_y, lab_x)

    # Sort tuple based on the length of the first entry in the tuple
    test_xy.sort(key=lambda t: len(t[0]))
    if lab_x:
        test_x, test_y, test_filename_y, test_lab_x = zip(*test_xy)
    else:
        test_x, test_y, test_filename_y = zip(*test_xy)

    return test_x, test_y, test_filename_y, test_lab_x

def sort_data_given_index(x, y, perm_index):
    assert len(x) == len(y)
    train_x = [None]* len(x)
    train_y = [None]* len(y)

    counter = 0
    for idx in perm_index:
        train_x[idx] = x[counter]
        train_y[idx] = y[counter]
        counter += 1

    return train_x, train_y

def split_data_into_chunks(x, y, batch_size, combine_y=True, lab_x=None):
    import keras.backend as K
    from keras.preprocessing import sequence

    test_x_chunks = list(chunks(x, batch_size))
    test_x = []
    test_x_len = 0

    test_y_chunks = list(chunks(y, batch_size))
    test_y = []
    test_y_len = 0

    assert len(test_x_chunks) == len(test_y_chunks)

    for i in range(len(test_x_chunks)):
        curr_test_x = test_x_chunks[i]
        curr_test_x = sequence.pad_sequences(curr_test_x)
        test_x.append(curr_test_x)
        test_x_len += len(curr_test_x)

        curr_test_y = test_y_chunks[i]
        curr_test_y = np.array(curr_test_y, dtype=K.floatx())
        test_y.append(curr_test_y)
        test_y_len += len(curr_test_y)

    assert test_x_len == test_y_len
    assert test_x_len == len(y)

    if (combine_y):
        test_y = np.array(y, dtype=K.floatx())

    return test_x, test_y

def sort_and_split_data_into_chunks(x, y, filename_y, batch_size, lab_x=None):
    '''Sort based on length of x
    Split test data into chunks of N (batch size) and pad them per chunk
    Faster processing because of localized padding
    '''
    test_lab_x = None
    test_x, test_y, test_filename_y, test_lab_x = sort_data(x, y, filename_y, lab_x=lab_x)

    test_x, test_y = split_data_into_chunks(test_x, test_y, batch_size, lab_x=test_lab_x)
    return test_x, test_y, test_filename_y, test_lab_x

def specificity_score(y_actual, y_hat):
    """Calculate the specificity score given ground truth and predicted class."""

    tps, fps, tns, fns = 0, 0, 0, 0
    y_hat_len = len(y_hat)

    for i in range(y_hat_len):
        if y_actual[i] == y_hat[i] == 1:
            tps += 1
    for i in range(y_hat_len):
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            fps += 1
    for i in range(y_hat_len):
        if y_actual[i] == y_hat[i] == 0:
            tns += 1
    for i in range(y_hat_len):
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            fns += 1

    specificity = float(tns) / float(tns + fps)
    return specificity

def get_binary_predictions(pred, threshold=0.5):
    '''
    Convert [0,1] real number predictions to binary 1 or 0 predictions
    Using 0.5 as its default threshold unless specified
    '''
    binary_pred = pred
    high_indices = binary_pred >= threshold
    binary_pred[high_indices] = 1
    low_indices = binary_pred < threshold
    binary_pred[low_indices] = 0

    return binary_pred

def compute_class_weight(train_y):
    '''
    Compute class weight given imbalanced training data
    '''
    import sklearn.utils.class_weight as scikit_class_weight

    class_list = list(set(train_y))
    class_weight_value = scikit_class_weight.compute_class_weight('balanced', class_list, train_y)
    class_weight = dict()

    # Initialize all classes in the dictionary with weight 1
    curr_max = np.max(class_list)
    for i in range(curr_max):
        class_weight[i] = 1

    # Build the dictionary using the weight obtained the scikit function
    for i in range(len(class_list)):
        class_weight[class_list[i]] = class_weight_value[i]

    logger.info('Class weight dictionary: ' + str(class_weight))
    return class_weight

def get_permutation_list(args, train_y):
    '''
    Produce a fixed list of permutation indices for training
    So that we dont depend on keras shuffling
    '''
    if (args.shuffle_path):
        logger.info('Loading shuffle permutation list from : %s' % args.shuffle_path)
        permutation_list = np.loadtxt(args.shuffle_path, dtype=int)
        logger.info('Loading shuffle permutation list completed!')
        return permutation_list
    
    shuffle_list_filename = "shuffle_permutation_list_len" + str(len(train_y)) + "_seed" + str(args.shuffle_seed) + ".txt"

    logger.info('Creating and saving shuffle permutation list to %s' % shuffle_list_filename)
    if args.shuffle_seed > 0:
        np.random.seed(args.shuffle_seed)

    permutation_list = []
    for ii in range(args.epochs*2):
        p = np.random.permutation(len(train_y))
        permutation_list.append(p)

    permutation_list = np.asarray(permutation_list, dtype=int)
    np.savetxt(args.out_dir_path + "/" + shuffle_list_filename, permutation_list, fmt='%d')

    logger.info('Creating and saving shuffle permutation list completed!')
    return permutation_list

