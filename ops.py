import skimage as ski
import numpy as np


class svd_op():
    """
    An svd_op object takes a forward matrix and a resolution res. It stores the resolution res, the matrices A, A_inv, and the singular value decomposition U,V,sigma.
    __call__ applies the forward operator to a 2-D element in R^(res x res), svd_op.adjoint applies the adjoint and svd_op.inverse applies the inverse to a 2-D element in R^(res x ?)    
    """
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
    """
    A reco_op object takes the singular vectore matrices U and V, a filter g and a resolution res and builds a reconstruction operator.
    reco_op.reconstruct applies the reconstruction operator to a 2-D element in R^(res x ?)    
    """
    def __init__(self, U, V, g, res):
        self.rec= U@((V*g).T)
        self.res = res
    def reconstruct(self, sino):
        sino_vec = sino.flatten()
        x_vec = self.rec@sino_vec
        return np.reshape(x_vec, (self.res,self.res))
