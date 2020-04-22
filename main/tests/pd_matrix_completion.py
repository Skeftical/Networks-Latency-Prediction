import unittest
from main.codebase.models.matrix_completion import PenaltyDecomposition 
import numpy as np

class PreProcessingTestCase(unittest.TestCase):
    def setUp(self):
        #non-missing rows
        self.I0 = np.loadtxt('/home/fotis/dev_projects/PD_Completion_MF/I0.mat',skiprows=7)
        #non-missing columns
        self.J0 = np.loadtxt('/home/fotis/dev_projects/PD_Completion_MF/J0.mat',skiprows=7)
        #entries of non-missing values
        self.data = np.loadtxt('/home/fotis/dev_projects/PD_Completion_MF/data.mat',skiprows=5)
        self.I0 = self.I0.astype('int')
        self.J0 = self.J0.astype('int')
        self.M = np.loadtxt('/home/fotis/dev_projects/PD_Completion_MF/M.mat',skiprows=5)
        self.I0 -= 1
        self.J0 -= 1 
        self.pd = PenaltyDecomposition()

    def test_correctness(self):
        X, rx, iters = self.pd.fit_transform(self.data, self.I0, self.J0)
        loss = np.linalg.norm(X-self.M)/np.linalg.norm(self.M)
        self.assertLessEqual(loss-2.1438029193860132e-05,1e-6,
                         'Incorrect loss')
        self.assertEqual(rx,5,'Incorrect rank')

if __name__ == '__main__':
    unittest.main()
