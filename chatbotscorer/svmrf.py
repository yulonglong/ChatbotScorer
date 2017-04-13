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
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


def run_svmrf(args, fold, train_x, train_y, test_x, test_y):
    ltrain_x, ltest_x = helper.get_bag_of_words(train_x, test_x)
    
    # np.savetxt(args.out_dir_path + '/preds/train_x_f' + str(fold) + '.txt', ltrain_x, fmt='%i')
    # np.savetxt(args.out_dir_path + '/preds/test_x_f' + str(fold) + '.txt', ltest_x, fmt='%i')
    
    train_y = np.array(train_y)
    test_y = np.array(test_y)
    
    if args.model_type == "svm":
        if args.label_type == 'mean':
            from sklearn.svm import SVR
            clf = SVR(kernel='sigmoid', degree=3, gamma='auto', coef0=0.0, tol=0.001, \
                C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, \
                max_iter=10000)
        else:
            from sklearn.svm import SVC
            clf = SVC(C=1.0, cache_size=2000, class_weight=None, coef0=0.0,
                decision_function_shape=None, degree=3, gamma='auto', kernel='sigmoid',
                max_iter=10000, probability=False, random_state=None, shrinking=True,
                tol=0.001, verbose=False)
        
    elif args.model_type == "rf":
        if args.label_type == 'mean':
            from sklearn.ensemble import RandomForestRegressor
            clf = RandomForestRegressor(n_estimators=50, criterion='mse', \
                max_depth=20, min_samples_split=2, min_samples_leaf=1, \
                min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, \
                min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1, \
                random_state=None, verbose=0, warm_start=False)
        else:
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=50, criterion='entropy', \
                max_depth=20, min_samples_split=2, min_samples_leaf=1,\
                min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,\
                min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1, \
                random_state=None, verbose=0, warm_start=False, class_weight=None)
    clf.fit(ltrain_x, train_y)
    
    test_pred = clf.predict(ltest_x)
    # logger.info(test_pred)

    if args.label_type == 'mean':
        test_correlation, test_p_value = pearsonr(test_y,test_pred)
        logger.info(
            '[TEST]  Correlation-coef: %.3f, p-value: %.5f' % (test_correlation, test_p_value)
        )
        logger.info('------------------------------------------------------------------------')
        return test_correlation, test_p_value
    else:
        binary_test_pred = helper.get_binary_predictions(test_pred)
        (test_tps, test_fps, test_fns, test_tns,
            test_recall, test_precision, test_specificity,
            test_f1, test_accuracy) = helper.confusion_matrix(test_y, binary_test_pred, label_type=args.label_type)
        logger.info(
            '[TEST]  F1: %.3f, Recall: %.3f, Precision: %.3f, Acc: %.5f' % (
                test_f1, test_recall, test_precision,
                test_accuracy)
        )
        logger.info('------------------------------------------------------------------------')
        return test_f1, test_accuracy
    
