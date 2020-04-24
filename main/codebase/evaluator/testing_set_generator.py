import numpy as np

np.random.seed(5)

def missing_values(X,missing_value_ratio=0.3):
    """
    args :
        X : ndarray matrix
        missing_value_ratio : ratio of values that will be missing
    returns :
        X_missing: matrix with values missing
    """
    sample = np.random.uniform(size=(X.shape))
    sample[sample<=missing_value_ratio] = np.nan
    X_missing = X*(sample*0+1)
    X_missing[np.diag_indices(X_missing.shape[0])] = 0
    return X_missing



class TestingSetGenerator():

    def get_matrix(self,initial_matrix=0,num=100):
        i=initial_matrix
        j=0
        while True:
            if j==num:
                return
            sprob = np.random.rand()
            if sprob>=0.5:
                j+=1
                yield self.matrices[i],i
            i= (i+1)%len(self.matrices)

    def load_matrices(self):
        self.matrices = []
        for i in range(1,689):
            f = "SeattleData_{}".format(i)
            m = np.loadtxt('/home/fotis/DATA/NETWORKS/MATRIX/NetLatency-Data-master/Seattle/{}'.format(f),delimiter='\t')
            

            self.matrices.append(m)

    def initialize_test_set(self):
        start = 0 
        if self.lags is not None:
            start = self.lags
        test_set = [m for m in self.get_matrix(start,self.test_set_size)]
        self.test_set, self.test_set_indices = [m[0] for m in test_set], [m[1] for m in test_set]
        self.test_set_missing = list(map(lambda X: missing_values(X,self.missing_value_ratio),self.test_set))

    def __init__(self, choice='Seattle', missing_value_ratio=0.3, test_set_size=5, lags=None):
        '''
        args :
            choice : str
                The choice of the data set to load
            missing_value_ratio : float
                The ratio of missing values for each matrix in the test set
            test_set_size : int
                The number of matrices to evaluated
            lags : int
                If using models that have time-series component then this parameter makes sure
                that no matrix at index (i<lags) is selected
        '''
        self.matrices = None
        self.missing_value_ratio = missing_value_ratio
        self.test_set_size = test_set_size
        self.lags = lags
        self.test_set = None
        self.test_set_missing = None
        self.test_set_indices = None

        self.load_matrices()
        self.initialize_test_set()
