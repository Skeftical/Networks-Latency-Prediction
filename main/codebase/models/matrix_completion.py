import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import spdiags,diags
from scipy.linalg import svd
from numpy.linalg import matrix_rank

class SimpleMF():

    def predict(self):
        return np.dot(self.F, self.X)

    def fit(self, M):
        m = M.shape[1] # size of columns
        n = M.shape[0] # size of rows

        self.X = np.random.uniform(size=(self.k,n))
        self.F = np.random.uniform(size=(m, self.k))

        losses = []
        ix1, ix2 = (~np.isnan(M)).nonzero()
        for _ in range(self.iterations):
            Omega = zip(ix1, ix2)
            for i,t in Omega:
                ei = M[i,t] - np.dot(self.F[i,:], self.X[:,t])
                self.F[i,:]+=self.gamma*(ei*self.X[:,t]-self.lambda_f*self.F[i,:])
                self.X[:,t]+=self.gamma*(ei*self.F[i,:]-self.lambda_x*self.X[:,t])
                loss = np.linalg.norm(M - np.dot(self.F,self.X),ord='fro') # loss with random matrices
                losses.append(loss)

        self.losses = losses
    def __init__(self,iterations=10,lambda_f=0.5, lambda_x=0.5, rank=10, gamma=0.01, **kwargs):
        self.k = rank
        self.iterations = iterations
        self.gamma = gamma
        self.X = None # np.random.uniform(size=(self.k, N))
        self.F = None # np.random.uniform(size=(M, self.k))
        self.lambda_f = lambda_f
        self.lambda_x = lambda_x
        self.losses = None
        for k,v in kwargs.items():
            setattr(self, k, v)

class PenaltyDecomposition():
    def proj(self,A,M,I,J,l,u,tau,k):
        '''
        args :

        '''
        n, p = A.shape
        q = max(M.shape)
        tmp=0
        for s in range(q):
            i,j = I[s], J[s]
            tmp += (M[s] - A[i,j])**2

        if tmp==0:
            t = 1
        else:
            tmp = np.sqrt(tmp)
            t = tau/tmp

        if k>= n*p - q:
            X = np.minimum(np.maximum(A,l),u)
            if t<1:
                for s in range(q):
                    i,j = I[s],J[s]
                    tmpM = M[s]
                    X[i,j] = (A[i,j]- tmpM)*t + tmpM

            return X
        dim = (n*p)
        val = np.zeros((dim,1))
        x = np.zeros((dim,1))
        row = np.zeros((dim,1)).astype('int')
        col = np.zeros((dim,1)).astype('int')
        m=0
        for j in range(p):
            for i in range(n):
                row[m], col[m] = i,j
                tmpA = A[i,j]
                tmpX = np.minimum(np.maximum(tmpA, l),u)
                x[m] = tmpX
                val[m] = (tmpX-tmpA)**2 - tmpA**2
                m+=1

        if t>=1:
            for s in range(q):
                i,j = I[s], J[s]
                m = j*n + i
                x[m] = A[i,j]
                val[m] = np.inf
        else:
            for s in range(q):
                i,j = I[s], J[s]
                m = j*n + i
                tmpM = M[s]
                x[m] = (A[i,j]- tmpM)*t + tmpM
                val[m] = np.inf
        L = np.argsort(val,axis=0).reshape(-1,)
        tmp = val[L]
        X = np.zeros((n,p))
        for s in range(k):
            m = L[s]
            i = row[m]
            j = col[m]
            X[i,j] = x[m]
        for s in range(q):
            i,j = I[s], J[s]
            m = j*n + i
            X[i,j] = X[m]
        return X

    def PD_completion(self,M,I,J,tau,l,u,k,eps,maxit):
        n = int(np.max(I))+1
        p = int(np.max(J))+1
        data = np.zeros((n,p))
        q = max(M.shape)

        for s in range(q):
            i, j = I[s], J[s]
            data[i,j] = M[s]

        scale = svds(data, k=1, which='LM', return_singular_vectors=False)
        scale = float(min(max(abs(scale),1),1000))
        M = M/scale
        X = data/scale
        tau = tau/scale

        if (l*u>0) or (k>=n*p-q):
            k = n*p -q
        rho = 1e-1
        iters = 1
        tol = 1e-3
        old_obj = np.inf
        #correct until here
        while True:
            best_obj = np.inf
            while True:
                # Solve Y
                Y = self.proj(X,M,I,J,l,u,tau,k)
                #Solve X
                U,d,V = svd(Y,full_matrices=False) #SVD in numpy returns d as a vector whereas in MATLAB d is Matrix
                L = (d**2 > 2/rho)
                U = U[:,L]
                d = d[L]
                V = V[L,:]
                r = np.sum(d>0)
                D = spdiags(d,0,r,r).toarray()

                X = U @ D @ V
                obj = r + rho*np.linalg.norm(X-Y,'fro')**2/2
                chg = np.abs(obj-old_obj)
                err = chg/max(obj,1)
                old_obj = obj
                iters+=1
                if err<=tol or iters>maxit:
                    res = np.linalg.norm(X-Y,'fro')/np.linalg.norm(M)
                    if best_obj - obj > (1e-6)*np.abs(best_obj) or (best_obj==np.inf and obj <best_obj):
                        best_obj = obj
                        best_X = X
                        best_res = res
                        best_Y =  Y
                        best_U = U
                        best_V = V
                        best_d = d
                        best_r = r
                        m=1

                    if m>= min(2, best_r) or rho>1e+6: #maximum value for m is 2
                        break;

                    tmpU = best_U[:, best_r-1:best_r]
                    tmpV = best_V[best_r-1:best_r,:]
                    tmpD = spdiags(best_d[best_r-1],0,tmpU.shape[1],tmpV.shape[0]).toarray()
                    X = best_X - (tmpU @ tmpD @ tmpV).astype('float')
                    m+=1

            if best_obj<obj:
                X = best_X
                res = best_res
            if (res<eps or iters>maxit):
                break;
            rho = min(5*rho, 1e9)
            tol = max(tol/5,5e-5)
        X = best_X*scale
        Y = best_Y*scale
        rX = matrix_rank(X)
        return X, rX, iters

    def fit(self, M):
        I0,J0 = (~np.isnan(M)).nonzero()
        data = M[~np.isnan(M)]
        k = data.shape[0]
        self.X, self.rx, self.iters = self.PD_completion(data,I0,J0,self.tau,self.l,self.u,k,self.eps,self.maxit)
    def predict(self):
       return self.X

    def fit_transform(self, data, I0, J0):
       k = data.shape[0]
       X, rx, iters = self.PD_completion(data,I0,J0,self.tau,self.l,self.u,k,self.eps,self.maxit)
       return X, rx, iters

    def __init__(self, tau=10, l=0.001, u=1, eps=1e-3, maxit=np.inf, **kwargs):
        self.tau = tau
        self.l = l
        self.u = u
        self.eps = eps
        self.maxit = maxit
        for k,v in kwargs.items():
            setattr(self, k, v)        
