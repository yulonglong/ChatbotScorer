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
parser.add_argument("-maxp", "--max-path", dest="max_path", type=str, metavar='<str>', required=True, help="The path to the output of the maximum case set")
parser.add_argument("-minp", "--min-path", dest="dev_path", type=str, metavar='<str>', required=True, help="The path to the output of the minimum case set")
parser.add_argument("-ref", "--reference", dest="ref_path", type=str, metavar='<str>', required=True, help="The path to the output of the averaging ground truth")

args = parser.parse_args()

for fold in range(10):
    test_max = np.loadtxt(args.max_path + 'test_pred_9999_f' + str(self.fold) + '.txt',dtype=float)
    test_min = np.loadtxt(args.min_path + 'test_pred_9999_f' + str(self.fold) + '.txt',dtype=float)
    test_ref = np.loadtxt(args.ref_path + 'test_ref_f' + str(self.fold) + '.txt',dtype=float)

    