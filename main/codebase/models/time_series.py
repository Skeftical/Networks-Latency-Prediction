import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing
from itertools import product
from scipy.interpolate import interp1d
from main.codebase.models.matrix_completion import SimpleMF
class SES():

    def __missing_value_imputation(self, y):
        '''
        args :
            y : vector with missing values
        '''
        tvals = np.arange(1, y.size+1)
        mean = np.mean(y[~np.isnan(y)])
        # If the first or last elements are missing impute by mean
        if np.isnan(y[0]):
            y[0] = mean
        if np.isnan(y[-1]):
            y[-1] = mean
        mask = ~np.isnan(y)
        f  = interp1d(tvals[mask],yvals[mask], kind='linear') #Could optimize kind
        return f(tvals)

    def fit(self, matrices, ix):
        '''
          args :
            matrices : list of matrices with missing entries
            ix : index of matrix to be predicted
        '''
        shape_of_matrix = matrices[0].shape
        past_matrices = matrices[ix-self.L:ix]
        past_history = np.array(past_matrices)
        ys_predicted = np.zeros(shape_of_matrix)
        iter_entries = list(product(range(shape_of_matrix[0]), range(shape_of_matrix[1])))
        impute_by_column = []
        for i,j in iter_entries:
            d = past_history[:,i,j]
            if np.isnan(d).sum()==d.size:
                #All past values are missing TS prediction cannot be performed
                # Impute value by column-value
                impute_by_column.append((i,j))
                continue;
            elif np.isnan(d).sum()>0:
                d = self.__missing_value_imputation(d)
            m = SimpleExpSmoothing(d).fit(smoothing_level=self.smoothing_level,optimized=self.optimized)
            y_hat = float(m.forecast(1))
            ys_predicted[i,j] = y_hat

        for i,j in impute_by_column:
            #Impute by both the column values and row columns
            ys_predicted[i,j] = np.mean(np.concatenate((ys_predicted[i,:], ys_predicted[:,j])))

        self.Mhat = ys_predicted

    def predict(self):
        return self.Mhat

    def __init__(self, lags=5, smoothing_level=0.2, optimized=False, **kwargs):
        self.smoothing_level = smoothing_level
        self.optimized = optimized
        self.L = lags
        self.Mhat = None
        for k,v in kwargs.items():
            setattr(self, k, v)

class TSMF():

    def fit(self, matrices, ix):
        self.mf.fit(matrices[ix])
        self.Mhat_MF = self.mf.predict()
        self.ts.fit(matrices, ix)
        self.Mhat_TS = self.ts.predict()
    def predict(self):
        return self.alpha*self.Mhat_MF + (1-self.alpha)*self.Mhat_TS

    def __init__(self, alpha=0.5, lags=5, smoothing_level=0.2, optimized=False,
                    iterations=10,lambda_f=0.5, lambda_x=0.5, rank=10, gamma=0.01, **kwargs):
        self.alpha = alpha
        self.lags = lags
        self.smoothing_level = smoothing_level
        self.optimized = optimized
        self.iterations = iterations
        self.lambda_f = lambda_f
        self.lambda_x = lambda_x
        self.rank = rank
        self.gamma = gamma
        for k,v in kwargs:
            setattr(self, k, v)
        self.mf = SimpleMF(iterations=self.iterations, lambda_f=self.lambda_f, lambda_x=self.lambda_x, rank=self.rank, gamma=0.01)
        self.ts = SES(lags=self.lags,smoothing_level=self.smoothing_level, optimized=self.optimized)
        self.Mhat_MF = None
        self.Mhat_TS = None
