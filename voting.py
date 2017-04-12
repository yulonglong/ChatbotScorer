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

###############################################################################################################################
## Parse arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("-maxp", "--max-path", dest="max_path", type=str, metavar='<str>', required=True, help="The path to the output of the maximum case set")
parser.add_argument("-minp", "--min-path", dest="min_path", type=str, metavar='<str>', required=True, help="The path to the output of the minimum case set")
parser.add_argument("-ref", "--reference", dest="ref_path", type=str, metavar='<str>', required=True, help="The path to the output of the averaging ground truth")
parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True, help="The path to the output directory")

args = parser.parse_args()

U.mkdir_p(args.out_dir_path)
U.set_logger(args.out_dir_path)
U.print_args(args)

total_max = np.array([])
total_min = np.array([])
total_mean = np.array([])

total_binary_max = np.array([])
total_binary_min = np.array([])
total_binary_mean = np.array([])

total_ref = np.array([])

total_max_pearson, total_min_pearson, total_mean_pearson = 0.0, 0.0, 0.0
total_binary_max_pearson, total_binary_min_pearson, total_binary_mean_pearson = 0.0, 0.0, 0.0

for fold in range(10):
    ###############################################################################
    ## Get original output and groundtruth (real-valued) and voting result
    #
    test_max = np.loadtxt(args.max_path + 'test_pred_9999_f' + str(fold) + '.txt',dtype=float)  
    test_min = np.loadtxt(args.min_path + 'test_pred_9999_f' + str(fold) + '.txt',dtype=float)
    test_ref = np.loadtxt(args.ref_path + 'test_ref_f' + str(fold) + '.txt',dtype=float)
    test_mean = np.mean([test_max, test_min], axis=0)

    ###############################################################################
    ## Get binary output and voting result
    #
    binary_test_max = helper.get_binary_predictions(test_max)
    binary_test_min = helper.get_binary_predictions(test_min)
    binary_test_mean = np.mean([binary_test_max, binary_test_min], axis=0)

    ###############################################################################
    ## Append output for aggregate Pearson coefficient over the 10 folds
    #
    total_max = np.append(total_max, test_max)
    total_min = np.append(total_min, test_min)
    total_mean = np.append(total_mean, test_mean)
    total_binary_max = np.append(total_binary_max, binary_test_max)
    total_binary_min = np.append(total_binary_min, binary_test_min)
    total_binary_mean = np.append(total_binary_mean, binary_test_mean)
    total_ref = np.append(total_ref, test_ref)

    logger.info("---------------- FOLD %i --------------------" % fold)
    # logger.info(test_max)
    # logger.info(binary_test_max)
    # logger.info(test_min)
    # logger.info(binary_test_min)
    # logger.info(test_mean)
    # logger.info(binary_test_mean)
    # logger.info(test_ref)

    max_pearson, _ = pearsonr(test_max, test_ref)
    min_pearson, _ = pearsonr(test_min, test_ref)
    mean_pearson, _ = pearsonr(test_mean, test_ref)

    binary_max_pearson, _ = pearsonr(binary_test_max, test_ref)
    binary_min_pearson, _ = pearsonr(binary_test_min, test_ref)
    binary_mean_pearson, _ = pearsonr(binary_test_mean, test_ref)

    total_max_pearson += max_pearson
    total_min_pearson += min_pearson
    total_mean_pearson += mean_pearson
    total_binary_max_pearson += binary_max_pearson
    total_binary_min_pearson += binary_min_pearson
    total_binary_mean_pearson += binary_mean_pearson
    logger.info("  REAL   :   MaxP: %.3f, MinP: %.3f, VotingP: %.3f " % (max_pearson, min_pearson, mean_pearson))
    logger.info("  BINARY :   MaxP: %.3f, MinP: %.3f, VotingP: %.3f " % (binary_max_pearson, binary_min_pearson, binary_mean_pearson))



logger.info("===================== OVERALL AVG =========================")
logger.info("  REAL   :   MaxP: %.3f, MinP: %.3f, VotingP: %.3f " % (total_max_pearson/10.0, total_min_pearson/10.0, total_mean_pearson/10.0))
logger.info("  BINARY :   MaxP: %.3f, MinP: %.3f, VotingP: %.3f " % (total_binary_max_pearson/10.0, total_binary_min_pearson/10.0, total_binary_mean_pearson/10.0))
logger.info("===========================================================")

logger.info("===================== OVERALL TOTAL =======================")
max_pearson, _ = pearsonr(total_max, total_ref)
min_pearson, _ = pearsonr(total_min, total_ref)
mean_pearson, _ = pearsonr(total_mean, total_ref)
binary_max_pearson, _ = pearsonr(total_binary_max, total_ref)
binary_min_pearson, _ = pearsonr(total_binary_min, total_ref)
binary_mean_pearson, _ = pearsonr(total_binary_mean, total_ref)
logger.info("  REAL   :   MaxP: %.3f, MinP: %.3f, VotingP: %.3f " % (max_pearson, min_pearson, mean_pearson))
logger.info("  BINARY :   MaxP: %.3f, MinP: %.3f, VotingP: %.3f " % (binary_max_pearson, binary_min_pearson, binary_mean_pearson))
logger.info("===========================================================")
