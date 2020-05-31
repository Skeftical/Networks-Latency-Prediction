import numpy as np
from .testing_set_generator import TestingSetGenerator
from main.codebase.models.euclidean import Vivaldi
from main.codebase.models.matrix_completion import SimpleMF, PenaltyDecomposition, NMFWrapper, SVDWrapper
from main.codebase.models.networks3d import Networks3D, Networks3DAlg2
from main.codebase.models.time_series import SES, TSMF, TSMFAbstract
from .config import *
from functools import reduce
import argparse
import logging
import os
import sys
import pandas as pd
from datetime import datetime
from time import time
from joblib import Parallel, delayed

np.random.seed(5)
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", "-v", dest='verbosity', help="increase output verbosity",
                    action="store_true")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-a',"--all",dest="test_all_models", action="store_true", help="Test on all models")
group.add_argument('-m','--models',dest='model_list', nargs='+', help='Models to evaluate')
parser.add_argument('-p', '--processes', dest='processes', type=int)
parser.add_argument("test_size", help="Size of test set to evaluate models on", type=int)
parser.add_argument("missing_value_ratio", help='Ratio of missing values in matrices',type=float)
parser.add_argument("fpath", help='Path to matrices', type=str)
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
   fileHandler = logging.FileHandler("{0}/eval-run{1}.log".format(logPath, datetime.now().isoformat()))
   fileHandler.setFormatter(logFormatter)
   logger.addHandler(fileHandler)
   handler = logging.StreamHandler(sys.stdout)
   handler.setFormatter(logFormatter)
   logger.addHandler(handler)
logger.info("Loading Testing Set Generator")
ts = TestingSetGenerator(fpath=args.fpath, missing_value_ratio=args.missing_value_ratio, test_set_size=args.test_size, lags=LAGS)


models = {}
models['TSMF'] = TSMF
models['SES'] = SES
models['TSMF-SVD'] = TSMFAbstract(SVDWrapper(rank=10),SES(**parameters['SES']))
models['TSMF-NMF'] = TSMFAbstract(NMFWrapper(rank=10),SES(**parameters['SES']))
models_set = list(models.keys())
if args.test_all_models:
    logger.info("Testing on all models")
elif args.model_list:
    for m in models_set:
        if m not in args.model_list:
            models.pop(m)


eval_df = {}
logger.info("Beginning evaluation on models :\n {}".format('\t'.join(models.keys())))
for i in range(len(ts.test_set)):
    logger.info("Run {}/{}".format(i+1, args.test_size))
    ix = ts.test_set_indices[i]
    logger.info("On matrix ID {}".format(ix))
    M = ts.test_set_missing[i]
    M_true = ts.test_set[i]
    if 'true' in eval_df:
        eval_df['true'] = np.concatenate([eval_df['true'], M_true.flatten()])
    else:
        eval_df['true'] = M_true.flatten()

def eval_on_model(model_label, i):
    ix = ts.test_set_indices[i]
    start = time()
    if model_label in parameters:
        model = models[model_label](**parameters[model_label])
    elif model_label not in ['SVD','NMF','TSMF-SVD', 'TSMF-NMF']:
        model = models[model_label]()
    else:
        model = models[model_label]

    M_hat = model.forecast(ts.matrices_with_missing, ix)
    logger.info("Evaluation completed on {}, took {}s".format(model_label,time()-start))
    return M_hat.flatten()

for model_label in models:
        logger.info("Starting on model {}".format(model_label))
        mhats = Parallel(n_jobs=args.processes,verbose=1)(delayed(eval_on_model)(model_label,i) for i in range(len(ts.test_set)))
        eval_df[model_label] = np.concatenate(mhats)


eval_df = pd.DataFrame(eval_df)
eval_df.to_csv('output/Accuracy/forecasting_run_{}-{}-{}-{}.csv'.format(datetime.now().isoformat(),args.test_size, args.missing_value_ratio,args.fpath.split('/')[-2]))
print(eval_df)
for k,v in parameters.items():
    logger.info("Model {}\nParameters:\n{}".format(k,'\n'.join(['{}\t{}'.format(label,val) for label, val in v.items()])))
