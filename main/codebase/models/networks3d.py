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

    def __init__(self,max_iter=3, d_vivaldi=3, gamma_vivaldi=0.01, iters_vivaldi=200,
                    tau=10, l=0.001, u=1, eps=1e-3, maxit=np.inf, **kwargs):
            # 10,0.001,1,1e-3,inf obtained from paper experiments
            self.vivaldi = Vivaldi(d_vivaldi, gamma_vivaldi, iters_vivaldi)
            self.pd = PenaltyDecomposition(tau, l, u, eps, maxit)
            self.max_iter = max_iter
            for k,v in kwargs.items():
                setattr(self, k, v)
class Networks3DAlg2():

        def __get_column_stacked(self, matrices):
            c = list(map(lambda x: x.flatten(), matrices))
            return np.column_stack(c[::-1])

        def __transform_to_indices(self, vector, shape):
            return list(map(lambda x: (x//shape[1], x%shape[0]), vector))

        def __transform_to_masks(self, theta, shape):
                idx = np.zeros(shape)
                for tup in theta:
                    idx[tup] = 1

        def __get_missing_thetas(self, Omega, shape):
            '''
            Returns two matrices, theta_a and theta_b which contain booleans that indicate
            the positions of elements that can be predicted using column_stacking or frame_stacking
            operations
            '''
            all_missing = np.sum(np.isnan(Omega),axis=1)==Omega.shape[1]
            theta_b = self.__transform_to_indices(np.argwhere(all_missing).reshape(-1,), shape)
            theta_a = self.__transform_to_indices(np.argwhere(~all_missing).reshape(-1,), shape)
            theta_b = self.__transform_to_masks(theta_b, shape)
            theta_a = self.__transform_to_masks(theta_a, shape)
            return (theta_a, theta_b)

        def __procedure2(self, matrices, ix):
            shape = matrices[0].shape
            Dks = [m for m in matrices]
            Dk_hats = [None for _ in range(len(matrices))]
            Fk_hats = [None for _ in range(len(matrices))]
            Fks = [None for _ in range(len(matrices))]
            for ITER in range(self.max_iter):
                print("Iter : {}/{}".format(ITER, self.max_iter))
                #### embedding process
                for i,Dk in enumerate(Dks):
                    self.vivaldi.fit(Dk)
                    Dk_hats[i] = self.vivaldi.predict()
                for i,Dk_hat in enumerate(Dk_hats):
                    Fks[i] = matrices[i]/Dk_hat
                    Fks[i] = np.where(Fks[i]==np.inf,1,Fks[i])
                frame_stacked = np.concatenate(Fks)
                print("Finished embedding process")
                #### MF process
                self.pd.fit(frame_stacked)
                frame_stacked_hat = self.pd.predict()
                for t in range(0,frame_stacked_hat.shape[0],shape[0]):
                    Fk_hat = frame_stacked_hat[t:t+shape[0],:]
                    assert(Fk_hat.shape==shape)
                    print(np.sum(Fk_hat==0))
                    Dks[t//shape[0]] = matrices[t//shape[0]]/Fk_hat
                    Dks[t//shape[0]] = np.where(Dks[t//shape[0]]==np.inf, 1, Dks[t//shape[0]])
                    assert(np.sum(Dks[t//shape[0]]==np.inf)==0)
                print("Finished MF process")

            Dk_hat = Dk_hats[ix]
            Fk_hat = Fk_hats[ix]
            return np.multiply(Dk_hat, Fk_hat)

        def fit(self, matrices, ix):
            '''
             args :
                matrices : list
                    A list of all matrices
                ix : int
                    Index  of the current matrix
            '''
            shape = matrices[0].shape
            #Procedure 1
            print("Beginning process 1")
            Omega = self.__get_column_stacked(matrices)
            theta_a, theta_b = self.__get_missing_thetas(Omega, shape)
            self.pd.fit(Omega)
            Omega_hat = self.pd.predict()
            M_c_hat = Omega_hat[:,ix].reshape(shape)
            #Procedure 2
            print("Beginning process 2")
            M_c_hat_proc2 = self.__procedure2(matrices, ix)
            Mhat = np.zeros(shape)
            Mhat[theta_a] = M_c_hat[theta_a]
            Mhat[theta_b] = M_c_hat_proc2[theta_b]
            self.Mhat = Mhat

        def predict(self):
            return self.Mhat

        def __init__(self,max_iter=3, d_vivaldi=3, gamma_vivaldi=0.01, iters_vivaldi=50,
                        tau=10, l=0.001, u=1, eps=1e-3, maxit=np.inf):
                # 10,0.001,1,1e-3,inf obtained from paper experiments
                self.vivaldi = Vivaldi(d_vivaldi, gamma_vivaldi, iters_vivaldi)
                self.pd = PenaltyDecomposition(tau, l, u, eps, maxit)
                self.max_iter = max_iter
