import numpy as np
from .testing_set_generator import TestingSetGenerator
import argparse
import logging
import os
import sys
import pandas as pd
from datetime import datetime
from time import time
from main.codebase.models.matrix_completion import SimpleMF
from itertools import product

np.random.seed(5)

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))

def loss(m_true, m_hat):
    return np.linalg.norm(m_true-m_hat)/np.linalg.norm(m_true)

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", "-v", dest='verbosity', help="increase output verbosity",
                    action="store_true")

hypertuned_models = {
    'SimpleMF' : {'iterations': np.arange(10,100,50),
                 'lambda_f':np.linspace(0,1,2), 'lambda_x': np.linspace(0,1,2),
                 'rank' : np.arange(5,30,20), 'gamma': np.linspace(0.01, 0.1, 2)
                 }
}

args = parser.parse_args()

logFormatter = logging.Formatter("[%(asctime)s] [%(levelname)-5.5s]  %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logPath = 'output/logs'
if not os.path.exists('output/hypertuning'):
        logger.info('creating directory hypertuning')
        os.makedirs('output/hypertuning')
if not os.path.exists('output/logs'):
        os.makedirs('output/logs')
if args.verbosity:
   print("verbosity turned on")
   fileHandler = logging.FileHandler("{0}/hypertuning-run{1}.log".format(logPath, datetime.now().isoformat()))
   fileHandler.setFormatter(logFormatter)
   logger.addHandler(fileHandler)
   handler = logging.StreamHandler(sys.stdout)
   handler.setFormatter(logFormatter)
   logger.addHandler(handler)
logger.info("Loading Testing Set Generator")

ts = TestingSetGenerator(missing_value_ratio=0.3, test_set_size=10, lags=10)

models = {}
models['SimpleMF'] = SimpleMF

best_params_df = {}
logger.info("Beginning hypertuning on models :\n {}".format('\t'.join(models.keys())))


for model_label in models:
    start = time()
    best_score = np.inf
    best_params = None
    for params in product_dict(**hypertuned_models[model_label]):
        errors = []
        for i in range(len(ts.test_set)):
            logger.info("Run {}/{}".format(i+1, len(ts.test_set)))
            ix = ts.test_set_indices[i]
            logger.info("On matrix ID {}".format(ix))
            M = ts.test_set_missing[i]
            M_true = ts.test_set[i]
            model = models[model_label](**params)
            model.fit(M)
            M_hat = model.predict()
            M_hat = np.where(np.isnan(M), M_hat, M_true)
            errors.append(loss(M_true, M_hat))
        error = np.mean(errors)
        if error<best_score:
            best_score = error
            best_params = params
    logger.info("Hypertuning completed on {}, took {}s".format(model_label,time()-start))
    logger.info("Best Score : {}\nBest Parameters {}".format(best_score, '\t'.join(['({},{})'.format(label,val) for label, val in best_params.items()])))
