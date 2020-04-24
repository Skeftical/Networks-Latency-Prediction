from main.codebase.models.euclidean import Vivaldi
from main.codebase.models.matrix_completion import PenaltyDecomposition
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
            self.pd.fit(Fk)
            Fk_hat = self.pd.predict()
            Dk = M/Fk_hat
        self.Dk_hat = Dk_hat
        self.Fk_hat = Fk_hat
    def predict(self):
        return np.multiply(self.Dk_hat, self.Fk_hat)

    def __init__(self,max_iter=5, d_vivaldi=3, gamma_vivaldi=0.01, iters_vivaldi=200,
                    tau=0, l=-np.inf, u=np.inf, eps=1e-5, maxit=100):
            self.vivaldi = Vivaldi(d_vivaldi, gamma_vivaldi, iters_vivaldi)
            self.pd = PenaltyDecomposition(tau, l, u, eps, maxit)
            self.max_iter = max_iter
