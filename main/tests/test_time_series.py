import unittest
from main.codebase.evaluator.testing_set_generator import TestingSetGenerator
from main.codebase.models.time_series import SES
import numpy as np

class EvaluatorSetTest(unittest.TestCase):
    def setUp(self):
        self.ts = TestingSetGenerator(missing_value_ratio=0.3, test_set_size=100, lags=10)
        self.ses = SES()

    def test_fit(self):
        self.ses.fit(self.ts.matrices_with_missing, self.ts.test_set_indices[0])
        self.assertEqual(self.ses.predict().shape,self.ts.matrices[0].shape)
    
if __name__ == '__main__':
    unittest.main()
