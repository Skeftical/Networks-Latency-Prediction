import unittest
from codebase.models.matrix_completion import PenaltyDecomposition
import numpy as np

class PreProcessingTestCase(unittest.TestCase):
    def setUp(self):
        #non-missing rows
        I0 = np.loadtxt('/home/fotis/dev_projects/PD_Completion_MF/I0.mat',skiprows=7)
        #non-missing columns
        J0 = np.loadtxt('/home/fotis/dev_projects/PD_Completion_MF/J0.mat',skiprows=7)
        #entries of non-missing values
        self.data = np.loadtxt('/home/fotis/dev_projects/PD_Completion_MF/data.mat',skiprows=5)
        self.I0 -= 1
        self.J0 -= 1
        self.pd = PenaltyDecomposition(tau, l, u, eps, maxit)

    def test_l1_cluster_dimensions(self):
        tau = 0
        l = -np.inf
        u = np.inf
        k = 800
        eps = 1e-5
        maxit = np.inf
        X, rx, iters = self.pd.PD_completion(self.data, self.I0, self.J0, tau, l, u, k , eps, maxit)
        loss = np.linalg.norm(X-M)/np.linalg.norm(M)
        self.assertEqual(loss, 2.1438029193860132e-05,
                         'Incorrect loss')
        self.assertEqual(rx, 5, 'Incorrect rank')

if __name__ == '__main__':
    unittest.main()
