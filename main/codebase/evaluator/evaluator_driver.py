import numpy as np
from testing_set_generator import TestingSetGenerator
from main.codebase.models.euclidean import Vivaldi
from main.codebase.models.matrix_completion import SimpleMF
import argparse
import logging
import os

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", "-v", dest='verbosity', help="increase output verbosity",
                    action="store_true")
parser.add_argument("--all",dest="test_all_models", action="store_true", help="Test on all models")
parser.add_argument("test_size", help="Size of test set to evaluate models on")
parser.add_argument("missing_value_ratio", help='Ratio of missing values in matrices')
args = parser.parse_args()

logger = logging.getLogger('main')
if args.verbosity:
   print("verbosity turned on")
   handler = logging.StreamHandler(sys.stdout)
   handler.setLevel(logging.DEBUG)
   logger.addHandler(handler)
if not os.path.exists('output/Accuracy'):
        logger.info('creating directory Accuracy')
        os.makedirs('output/Accuracy')

ts = TestingSetGenerator(missing_value_ratio=0.3, test_set_size=5)
