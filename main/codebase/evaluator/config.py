import numpy as np

simple_MF_params = {'iterations':10,'lambda_f':0.5, 'lambda_x':0.5, 'rank':10, 'gamma':0.01}
vivaldi_params = {'d':3,'gamma':0.01, 'iters':100}
pd_params = {'tau':10, 'l':0.001, 'u':1, 'eps':1e-3, 'maxit':np.inf}
networks3d_params = {'max_iter':3, 'd_vivaldi':3, 'gamma_vivaldi':0.01, 'iters_vivaldi':50,
                'tau':10, 'l':0.001, 'u':1, 'eps':1e-3, 'maxit':np.inf}

parameters = {}
parameters['SimpleMF'] = simple_MF_params
parameters['Vivaldi'] = vivaldi_params
parameters['PenaltyDecomposition'] = pd_params
parameters['Networks3D'] = networks3d_params
parameters['Networks3DAlg2'] = networks3d_params
