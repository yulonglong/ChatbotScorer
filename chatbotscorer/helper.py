import argparse
import logging
import numpy as np
from time import time
import sys
import utils as U
import pickle as pk

logger = logging.getLogger(__name__)

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

def confusion_matrix(y_actual, y_hat):
    """Calculate the confusion matrix given ground truth and predicted class."""

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

    return tps, fps, fns, tns

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
