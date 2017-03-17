"""Evaluator.py - A python class/module to calculate neural network performance."""

import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import helper as helper

####################################################################################################
## Evaluator class
#

class Evaluator(object):
    """Evaluator class - A python class/module to calculate neural network performance."""

    def __init__(self, logger, out_dir, train, dev, test, model_type, batch_size_eval=256):
        """Initialized Evaluator class"""
        self.logger = logger
        self.out_dir = out_dir
        self.model_type = model_type
        self.batch_size_eval = batch_size_eval

        self.train_x, self.train_y = (train[0], train[1])
        self.dev_x, self.dev_y = (dev[0], dev[1])
        self.test_x, self.test_y = (test[0], test[1])

        self.train_y_org = self.train_y.astype('int32')
        self.dev_y_org = self.dev_y.astype('int32')
        self.test_y_org = self.test_y.astype('int32')

        self.best_dev = [-1, -1, -1, -1]
        self.best_test = [-1, -1, -1, -1]
        self.best_dev_epoch = -1
        self.best_test_missed = -1
        self.best_test_missed_epoch = -1
        self.dump_ref_scores()

        self.dev_loss, self.dev_metric = 0.0, 0.0
        self.test_loss, self.test_metric = 0.0, 0.0

        self.train_recall = 0.0
        self.train_precision = 0.0
        self.train_f1 = 0.0
        self.train_specificity = 0.0
        self.train_accuracy = 0.0

        self.dev_recall = 0.0
        self.dev_precision = 0.0
        self.dev_f1 = 0.0
        self.dev_specificity = 0.0

        self.test_recall = 0.0
        self.test_precision = 0.0
        self.test_f1 = 0.0
        self.test_specificity = 0.0

        self.train_pred = np.array([])
        self.dev_pred = np.array([])
        self.test_pred = np.array([])

    def dump_ref_scores(self):
        """Dump reference (ground truth) scores"""
        np.savetxt(self.out_dir + '/preds/train_ref.txt', self.train_y_org, fmt='%i')
        np.savetxt(self.out_dir + '/preds/dev_ref.txt', self.dev_y_org, fmt='%i')
        np.savetxt(self.out_dir + '/preds/test_ref.txt', self.test_y_org, fmt='%i')

    def dump_train_predictions(self, train_pred, epoch):
        """Dump predictions of the model on the training set"""
        np.savetxt(self.out_dir + '/preds/train_pred_' + str(epoch) + '.txt',
                   train_pred, fmt='%.8f')

    def dump_predictions(self, dev_pred, test_pred, epoch):
        """Dump predictions of the model on the dev and test set"""
        np.savetxt(self.out_dir + '/preds/dev_pred_' + str(epoch) + '.txt', dev_pred, fmt='%.8f')
        np.savetxt(self.out_dir + '/preds/test_pred_' + str(epoch) + '.txt', test_pred, fmt='%.8f')

    def evaluate(self, model, epoch):
        """Evaluate on dev and test set, given a trained model at a given epoch"""

        # Reset train_pred, dev_pred and test_pred
        self.train_pred = np.array([])
        self.dev_pred = np.array([])
        self.test_pred = np.array([])

        curr_train_pred = model.predict(
            self.train_x, batch_size=self.batch_size_eval).squeeze()
        self.train_pred = np.append(self.train_pred, curr_train_pred)

        curr_dev_pred = model.predict(
            self.dev_x, batch_size=self.batch_size_eval).squeeze()
        self.dev_pred = np.append(self.dev_pred, curr_dev_pred)

        curr_test_pred = model.predict(
            self.test_x, batch_size=self.batch_size_eval).squeeze()
        self.test_pred = np.append(self.test_pred, curr_test_pred)

        # self.dump_train_predictions(self.train_pred)
        self.dump_predictions(self.dev_pred, self.test_pred, epoch)

        binary_train_pred = helper.get_binary_predictions(self.train_pred)
        self.train_recall = recall_score(self.train_y_org, binary_train_pred)
        self.train_precision = precision_score(self.train_y_org, binary_train_pred)
        self.train_f1 = f1_score(self.train_y_org, binary_train_pred)
        self.train_specificity = helper.specificity_score(self.train_y_org, binary_train_pred)
        self.train_accuracy = accuracy_score(self.train_y_org, binary_train_pred)

        binary_dev_pred = helper.get_binary_predictions(self.dev_pred)
        self.dev_recall = recall_score(self.dev_y_org, binary_dev_pred)
        self.dev_precision = precision_score(self.dev_y_org, binary_dev_pred)
        self.dev_f1 = f1_score(self.dev_y_org, binary_dev_pred)
        self.dev_specificity = helper.specificity_score(self.dev_y_org, binary_dev_pred)

        binary_test_pred = helper.get_binary_predictions(self.test_pred)
        self.test_recall = recall_score(self.test_y_org, binary_test_pred)
        self.test_precision = precision_score(self.test_y_org, binary_test_pred)
        self.test_f1 = f1_score(self.test_y_org, binary_test_pred)
        self.test_specificity = helper.specificity_score(self.test_y_org, binary_test_pred)

        if self.dev_f1 > self.best_dev[0]:
            self.best_dev = [self.dev_f1, self.dev_recall,
                             self.dev_precision, self.dev_specificity]
            self.best_test = [self.test_f1, self.test_recall,
                              self.test_precision, self.test_specificity]
            self.best_dev_epoch = epoch
            model.save_weights(self.out_dir + '/models/best_model_weights.h5', overwrite=True)

        if self.test_f1 > self.best_test_missed:
            self.best_test_missed = self.test_f1
            self.best_test_missed_epoch = epoch

    def print_info(self):
        """Print information on the current performance of the model"""

        self.logger.info('[TRAIN] F1: %.3f, Recall: %.3f, Precision: %.3f, Spec: %.5f, Acc: %.5f' % (
        	self.train_f1, self.train_recall, self.train_precision, self.train_specificity, self.train_accuracy))

        self.logger.info(
            '[DEV]   F1: %.3f, Recall: %.3f, Precision: %.3f, Spec: %.5f (Best @ %i: {{%.3f}}, %.3f, %.3f, %.5f)' % (
                self.dev_f1, self.dev_recall, self.dev_precision,
                self.dev_specificity, self.best_dev_epoch,
                self.best_dev[0], self.best_dev[1], self.best_dev[2], self.best_dev[3])
        )
        self.logger.info(
            '[TEST]  F1: %.3f, Recall: %.3f, Precision: %.3f, Spec: %.5f (Best @ %i: {{%.3f}}, %.3f, %.3f, %.5f)' % (
                self.test_f1, self.test_recall, self.test_precision,
                self.test_specificity, self.best_dev_epoch,
                self.best_test[0], self.best_test[1], self.best_test[2], self.best_test[3])
        )
        self.logger.info('------------------------------------------------------------------------')
        return self.best_test[0]
