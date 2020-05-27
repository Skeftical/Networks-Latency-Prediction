import unittest
from main.codebase.evaluator.testing_set_generator import TestingSetGenerator
from main.codebase.models.time_series import SES, TSMF, TSMFAbstract
from main.codebase.models.matrix_completion import  NMFWrapper, SVDWrapper
import numpy as np

class EvaluatorSetTest(unittest.TestCase):
    def setUp(self):
        self.ts = TestingSetGenerator(missing_value_ratio=0.3, test_set_size=100, lags=10)
        self.ses = SES()
        self.tsmf = TSMF()

    def test_fit(self):
        self.ses.fit(self.ts.matrices_with_missing, self.ts.test_set_indices[0])
        self.assertEqual(self.ses.predict().shape,self.ts.matrices[0].shape)

    def test_fit_tsmf(self):
        self.tsmf.fit(self.ts.matrices_with_missing, self.ts.test_set_indices[0])
        self.assertEqual(self.tsmf.predict().shape,self.ts.matrices[0].shape)

    def test_initialize_values_with_dict(self):
        d = {'alpha':0.1, 'lags':10, 'smoothing_level':0.3, 'optimized':True,
                        'iterations':1,'lambda_f':0.1, 'lambda_x':0.7, 'rank':5, 'gamma':0.001}
        tsmf = TSMF(**d)
        self.assertEqual(tsmf.alpha, d['alpha'])
        self.assertEqual(tsmf.lags, d['lags'])
        self.assertEqual(tsmf.smoothing_level, d['smoothing_level'])
        self.assertEqual(tsmf.optimized, d['optimized'])
        self.assertEqual(tsmf.iterations, d['iterations'])
        self.assertEqual(tsmf.lambda_f, d['lambda_f'])
        self.assertEqual(tsmf.lambda_x, d['lambda_x'])
        self.assertEqual(tsmf.rank, d['rank'])
        self.assertEqual(tsmf.gamma, d['gamma'])

    def test_tsmf_abstract(self):
        nmf = NMFWrapper(rank=10)
        svd = SVDWrapper(rank=10)
        tsmfabs = TSMFAbstract(nmf,ses)
        tsmfabs.fit(self.ts.matrices_with_missing, self.ts.test_set_indices[0])
        tsmfabs = TSMFAbstract(svd,ses)
        tsmfabs.fit(self.ts.matrices_with_missing, self.ts.test_set_indices[0])

if __name__ == '__main__':
    unittest.main()
