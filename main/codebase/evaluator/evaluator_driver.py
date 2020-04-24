import numpy as np
from .testing_set_generator import TestingSetGenerator
from main.codebase.models.euclidean import Vivaldi
from main.codebase.models.matrix_completion import SimpleMF
import argparse
import logging
import os
np.random.seed(5)
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", "-v", dest='verbosity', help="increase output verbosity",
                    action="store_true")
parser.add_argument("--all",dest="test_all_models", action="store_true", help="Test on all models")
parser.add_argument("test_size", help="Size of test set to evaluate models on", type=int)
parser.add_argument("missing_value_ratio", help='Ratio of missing values in matrices',type=float)
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
logger.info("Loading Testing Set Generator")
ts = TestingSetGenerator(missing_value_ratio=args.missing_value_ratio, test_set_size=args.test_size)

def get_true_results(M, M_true):
    y_test = M_true[np.isnan(M)]
    y_test = y_test[y_test.nonzero()]
    return y_test

def get_results(M, M_hat):
    y_hat = M_hat[np.isnan(M)]
    y_hat = y_hat[y_test.nonzero()] #zero values crash relative error calculation
    return y_hat


models = {}
models['SimpleMF'] = SimpleMF
models['Vivaldi'] = Vivaldi
eval_df = {}
logger.info("Beginning evaluation on models :\n {}".format('\t'.join(models.keys())))
for ix in ts.test_set_indices:
    M = ts.test_set_missing[ix]
    M_true = ts.test_set[ix]
    if 'true' in eval_df:
        eval_df['true'] = np.concatenate([eval_df['true'], get_true_results(M, M_true)])
    else:
        eval_df['true'] = get_true_results(M, M_true)
    for model_label in models:
        model = models[model_label]()
        model.fit(M)
        M_hat = model.predict()
        if model_label in eval_df:
            eval_df[model_label] = np.concatenate([eval_df[model_label], get_results(M, M_hat)])
        else:
            eval_df[model_label] = get_results(M, M_hat)
    break;
print(eval_df)
