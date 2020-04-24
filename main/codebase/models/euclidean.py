from sklearn.metrics import pairwise_distances
import numpy as np

class Vivaldi():

    def _unit(self,v):
        return v/np.linalg.norm(v)

    def fit(self, M):
        '''
        args :
            M : ndarray - matrix with possibly missing values to approximate
        '''
        self.X = np.random.uniform(size=(M.shape[0],self.d))
        self._compute_coordinates(M)

    def predict(self):
        return pairwise_distances(self.X, metric='l2')

    def compute_error(self, M):
        X = self.predict()
        return np.linalg.norm(X-M,'fro')/np.linalg.norm(M,'fro')

    def _compute_coordinates(self, M):
        losses = []
        for _ in range(self.iters):
            for i in range(self.X.shape[0]):
                    F = 0
                    for j in range(self.X.shape[0]):
                        if i==j or np.isnan(M[i,j]):
                            continue;
                        e = M[i,j]- np.linalg.norm(self.X[i,:]-self.X[j,:])
                        F+= e*self._unit(self.X[i,:]-self.X[j,:])
                    self.X[i,:]+=self.gamma*F
            losses.append(self.compute_error(M))
        self.losses = losses

    def __init__(self,d=3,gamma=0.01, iters=100):
        self.X = None
        self.d = d
        self.gamma = gamma
        self.iters = iters
        self.losses = None
