from euclidean import Vivaldi
from matrix_completion import PenaltyDecomposition
import numpy as np


class Networks3D():

    def fit(self, M):
        Dk = M
        for _ in range(self.max_iter):
            #### embedding process
            self.vivaldi.fit(Dk)
            Dk_hat = self.vivaldi.predict()
            Fk = M/Dk_hat
            #### MF process
            self.pd.fit(M)
            Fk_hat = self.pd.predict()
            Dk = M/Fk_hat
        self.Dk_hat = Dk_hat
        self.Fk_hat = Fk_hat
    def predict(self):
        return np.multiply(self.Dk_hat, self.Fk_hat)

    def __init__(self,max_iter=20, d_vivaldi=2, gamma_vivaldi=0.01, iters_vivaldi=100,
                    tau=0, l=-np.inf, u=np.inf, eps=1e-5, maxit=np.inf):
            self.vivaldi = Vivaldi(d_vivaldi, gamma_vivaldi, iters_vivaldi)
            self.pd = PenaltyDecomposition(tau, l, u, eps, maxit)
