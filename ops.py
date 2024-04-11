#%%
import skimage as ski
import numpy as np


class svd_op():
    def __init__(self, A, res):
        self.A = A
        V, S, Ut = np.linalg.svd(self.A, full_matrices = False)
        self.U = Ut.T
        self.V = V
        self.sigma = S
        self.A_inv=self.U@((V/S).T)
        self.res = res

    def __call__(self, x):
        x_vec = x.flatten()
        sino_vec = self.A@x_vec
        return np.reshape(sino_vec,(self.res,-1))
    
    def adjoint(self, sino):
        sino_vec = sino.flatten()
        x_vec = self.A.T@sino_vec
        return np.reshape(x_vec, (self.res,self.res))
    
    def inverse(self, sino):
        sino_vec = sino.flatten()
        x_vec = self.A_inv@sino_vec
        return np.reshape(x_vec, (self.res,self.res))
    
class reco_op():
    def __init__(self, U, V, g, res):
        self.rec= U@((V*g).T)
        self.res = res
    def reconstruct(self, sino):
        sino_vec = sino.flatten()
        x_vec = self.rec@sino_vec
        return np.reshape(x_vec, (self.res,self.res))
# %%
