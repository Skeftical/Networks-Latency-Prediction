import unittest
from main.codebase.evaluator.testing_set_generator import TestingSetGenerator
from main.codebase.models.networks3d import Networks3DAlg2
import numpy as np

class EvaluatorSetTest(unittest.TestCase):
    def setUp(self):
        self.ts = TestingSetGenerator(missing_value_ratio=0.3, test_set_size=100, lags=10)
        self.N3D2 = Networks3DAlg2()

    def test_fit(self):
        self.N3D2.fit(self.ts.matrices_with_missing,0)

    def test_shape_of_mhat(self):
        self.assertEqual(self.N3D2.predict().shape,self.ts.matrices[0].shape)

if __name__ == '__main__':
    unittest.main()
