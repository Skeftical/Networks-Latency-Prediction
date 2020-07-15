import numpy as np
from .testing_set_generator import TestingSetGenerator
import argparse
import logging
import os
import sys
import pandas as pd
from datetime import datetime
from time import time
from main.codebase.models.time_series import TSMF
from itertools import product
from joblib import Parallel, delayed
from .config import *
np.random.seed(5)

def loss(m_true, m_hat):
    return np.linalg.norm(m_true-m_hat)/np.linalg.norm(m_true)

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", "-v", dest='verbosity', help="increase output verbosity",
                    action="store_true")

sensitivity = {'lags': np.arange(5,15,5), 'alpha':np.linspace(0,1,3),
                 'rank' : np.arange(5,15,5)
            }

args = parser.parse_args()

logFormatter = logging.Formatter("[%(asctime)s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logPath = 'output/logs'
if not os.path.exists('output/Sensitivity'):
        logger.info('creating directory Sensitivity')
        os.makedirs('output/Sensitivity')
if not os.path.exists('output/logs'):
        os.makedirs('output/logs')
if args.verbosity:
   print("verbosity turned on")
   fileHandler = logging.FileHandler("{0}/sensitivity-run{1}.log".format(logPath, datetime.now().isoformat()))
   fileHandler.setFormatter(logFormatter)
   logger.addHandler(fileHandler)
   handler = logging.StreamHandler(sys.stdout)
   handler.setFormatter(logFormatter)
   logger.addHandler(handler)

def evaluate_on_param(param, val):
    errors = []
    for i in range(len(ts.test_set)):
        logger.info("Run {}/{}".format(i+1, len(ts.test_set)))
        ix = ts.test_set_indices[i]
        logger.info("On matrix ID {}".format(ix))
        M = ts.test_set_missing[i]
        M_true = ts.test_set[i]
        if param=='lags':
            parameters['TSMF']['lags'] = val
        for _ in range(10):
            model = TSMF(**parameters['TSMF'])
            model.fit(ts.matrices_with_missing, ix)
            M_hat = model.predict()
            M_hat = np.where(np.isnan(M), M_hat, M_true)
            errors.append(loss(M_true, M_hat))

    return errors
eval_df = {}
eval_df['parameter'] = []
eval_df['error'] = []
eval_df['val'] = []
lag = 10

for param,vals in sensitivity.items():
    logger.info("Sensitivity analysis on {}".format(param))
    logger.info("Loading Testing Set Generator")
    if param=='lags':
        lag = np.max(vals)
    ts = TestingSetGenerator('/home/fotis/DATA/NETWORKS/MATRIX/NetLatency-Data-master/Seattle',missing_value_ratio=0.3, test_set_size=10, lags=lag)
    oerrors = Parallel(n_jobs=4,verbose=1)(delayed(evaluate_on_param)(param,val) for val in vals)
    noerrors = np.concatenate(oerrors)
    eval_df['error']+= noerrors.tolist()
    eval_df['parameter']+= [param for _ in range(noerrors.shape[0])]
    eval_df['val'] += [vals[i/vals.shape[0]] for i in range(noerrors.shape[0])]

eval_df = pd.DataFrame(eval_df)
eval_df.to_csv('output/Sensitivity/sensitivity.csv')
print(eval_df)
