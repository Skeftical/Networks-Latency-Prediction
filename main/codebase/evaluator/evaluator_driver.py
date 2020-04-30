import numpy as np
from .testing_set_generator import TestingSetGenerator
from main.codebase.models.euclidean import Vivaldi
from main.codebase.models.matrix_completion import SimpleMF, PenaltyDecomposition
from main.codebase.models.networks3d import Networks3D, Networks3DAlg2
from .config import *
import argparse
import logging
import os
import sys
import pandas as pd
from datetime import datetime
np.random.seed(5)
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", "-v", dest='verbosity', help="increase output verbosity",
                    action="store_true")
parser.add_argument("--all",dest="test_all_models", action="store_true", help="Test on all models")
parser.add_argument("test_size", help="Size of test set to evaluate models on", type=int)
parser.add_argument("missing_value_ratio", help='Ratio of missing values in matrices',type=float)
args = parser.parse_args()

logFormatter = logging.Formatter("[%(asctime)s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logPath = 'output/logs'
if not os.path.exists('output/Accuracy'):
        logger.info('creating directory Accuracy')
        os.makedirs('output/Accuracy')
if not os.path.exists('output/logs'):
        os.makedirs('output/logs')
if args.verbosity:
   print("verbosity turned on")
#   fileHandler = logging.FileHandler("{0}/eval-run{1}.log".format(logPath, datetime.now().isoformat()))
#   fileHandler.setFormatter(logFormatter)
#   logger.addHandler(fileHandler)
   handler = logging.StreamHandler(sys.stdout)
   handler.setFormatter(logFormatter)
   logger.addHandler(handler)
logger.info("Loading Testing Set Generator")
ts = TestingSetGenerator(missing_value_ratio=args.missing_value_ratio, test_set_size=args.test_size)

def get_true_results(M, M_true):
    y_test = M_true[np.isnan(M)]
    y_test = y_test[y_test.nonzero()]
    return y_test

def get_results(M, M_true, M_hat):
    y_hat = M_hat[np.isnan(M)]
    y_test = M_true[np.isnan(M)]
    y_hat = y_hat[y_test.nonzero()] #zero values crash relative error calculation
    return y_hat


models = {}
models['SimpleMF'] = SimpleMF
models['Vivaldi'] = Vivaldi
models['PenaltyDecomposition'] = PenaltyDecomposition
models['Networks3D'] = Networks3D
models['Networks3DAlg2'] = Networks3DAlg2
eval_df = {}
logger.info("Beginning evaluation on models :\n {}".format('\t'.join(models.keys())))
for i in range(len(ts.test_set)):
    logger.info("Run {}/{}".format(i+1, args.test_size))
    ix = ts.test_set_indices[i]
    M = ts.test_set_missing[i]
    M_true = ts.test_set[i]
    if 'true' in eval_df:
        eval_df['true'] = np.concatenate([eval_df['true'], get_true_results(M, M_true)])
    else:
        eval_df['true'] = get_true_results(M, M_true)
    for model_label in models:
        if model_label in parameters:
            model = models[model_label](**parameters[model_label])
        else:
            model = models[model_label]()
        if model_label=='Networks3DAlg2':
            model.fit(ts.matrices_with_missing[:ix+1])
        else:
            model.fit(M)
        M_hat = model.predict()
        if model_label in eval_df:
            eval_df[model_label] = np.concatenate([eval_df[model_label], get_results(M, M_true, M_hat)])
        else:
            eval_df[model_label] = get_results(M, M_true, M_hat)

eval_df = pd.DataFrame(eval_df)
eval_df.to_csv('output/Accuracy/evaluation_run_{}.csv'.format(datetime.now().isoformat()))
print(eval_df)
for k,v in parameters.items():
    logger.info("Model {}\nParameters:\n{}".format(k,'\n'.join(['{}\t{}'.format(label,val) for label, val in v.items()])))
