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

    def load_matrices(self, fpath):
        self.matrices = []
        dir = os.listdir(fpath)
        fname = dir[0].split('_')[0]
        no_files = len(dir)
        for i in range(1,no_files+1):
            f = "{}_{}".format(fname, i)
            m = np.loadtxt('{}/{}'.format(fpath, f),delimiter='\t')
            self.matrices.append(m)


    def initialize_test_set(self):
        start = 0
        if self.lags is not None:
            start = self.lags
        test_set = [m for m in self.get_matrix(start,self.test_set_size)]
        self.test_set, self.test_set_indices = [m[0] for m in test_set], [m[1] for m in test_set]
        self.matrices_with_missing = list(map(lambda X: missing_values(X,self.missing_value_ratio),self.matrices))
        self.test_set_missing = [self.matrices_with_missing[i] for i in self.test_set_indices]

    def __init__(self, fpath, missing_value_ratio=0.3, test_set_size=5, lags=None, hypertuning_set=None):
        '''
        args :
            fpath : str
                File path to list of matrices. Matrices need to start with a name followed by the sequence number [1, N] eg. 'Seattle_1'
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
        self.hypertuning_set = hypertuning_set
        self.test_set = None
        self.test_set_missing = None
        self.matrices_with_missing = None
        self.test_set_indices = None

        self.load_matrices(fpath)
        self.initialize_test_set()
