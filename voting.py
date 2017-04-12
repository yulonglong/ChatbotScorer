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
total_ref = np.array([])

total_max_pearson, total_min_pearson, total_mean_pearson = 0.0, 0.0, 0.0

for fold in range(10):
    test_max = np.loadtxt(args.max_path + 'test_pred_9999_f' + str(fold) + '.txt',dtype=float)  
    test_min = np.loadtxt(args.min_path + 'test_pred_9999_f' + str(fold) + '.txt',dtype=float)
    test_ref = np.loadtxt(args.ref_path + 'test_ref_f' + str(fold) + '.txt',dtype=float)
    test_mean = np.mean([test_max, test_min], axis=0)

    total_max = np.append(total_max, test_max)
    total_min = np.append(total_min, test_min)
    total_mean = np.append(total_mean, test_mean)
    total_ref = np.append(total_ref, test_ref)

    logger.info("---------------- FOLD %i --------------------" % fold)
    # logger.info(test_max)
    # logger.info(test_min)
    # logger.info(test_mean)
    # logger.info(test_ref)
    max_pearson, _ = pearsonr(test_max, test_ref)
    min_pearson, _ = pearsonr(test_min, test_ref)
    mean_pearson, _ = pearsonr(test_mean, test_ref)
    total_max_pearson += max_pearson
    total_min_pearson += min_pearson
    total_mean_pearson += mean_pearson
    logger.info(" MaxP: %.3f, MinP: %.3f, VotingP: %.3f " % (max_pearson, min_pearson, mean_pearson))



logger.info("===================== OVERALL AVG =========================")
logger.info(" MaxP: %.3f, MinP: %.3f, VotingP: %.3f " % (total_max_pearson/10.0, total_min_pearson/10.0, total_mean_pearson/10.0))
logger.info("===========================================================")

logger.info("===================== OVERALL TOTAL =======================")
max_pearson, _ = pearsonr(total_max, total_ref)
min_pearson, _ = pearsonr(total_min, total_ref)
mean_pearson, _ = pearsonr(total_mean, total_ref)
logger.info(" MaxP: %.3f, MinP: %.3f, VotingP: %.3f " % (max_pearson, min_pearson, mean_pearson))
logger.info("===========================================================")
