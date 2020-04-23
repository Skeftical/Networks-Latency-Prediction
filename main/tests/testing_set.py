import unittest
from main.codebase.evaluator.testing_set_generator import TestingSetGenerator
import numpy as np

class EvaluatorSetTest(unittest.TestCase):
    def setUp(self):
        self.ts = TestingSetGenerator(missing_value_ratio=0.5, test_set_size=100, lags=10)

    def test_length_of_arrays(self):
        self.assertEqual(len(self.ts.test_set),100)
        self.assertEqual(len(self.ts.test_set_missing),100)
        self.assertEqual(len(self.ts.test_set_indices),100)

    def test_lags(self):
        x = np.min(self.ts.test_set_indices)
        self.assertTrue(x>self.ts.lags,
                            'Minimum provided index is {}'.format(x))

    def test_missing_value_correctness(self):
        ratio = np.array([np.isnan(m)/m.size for m in self.ts.test_set_missing])
        ratio = np.logical_and(ratio>=(0.5-1e-5), ratio<=(0.5+1e5))
        self.assertEqual(np.sum(ratio), 100, 'Ratios {}'.format(ratio))

if __name__ == '__main__':
    unittest.main()
