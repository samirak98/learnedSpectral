import numpy as np
from ops import svd_op

def get_pi(img_data, operator):
    """
    Parameters
    ----------
    img_data : np.array
        2-D array of shape num_samples x res*res
    operator: svd_op
        svd_op.U has to be of shape res*res x res*res
        svd_op.res has to be res

    Returns
    -------
    np.array
    
    The coefficients Pi_n that correspond to img_data and the singular vectors svd_op.U
    """
    num_samples = img_data.shape[-1]
    return np.sum((svd_op.U.T@img_data)**2, axis = 1)/num_samples
    
def get_delta(noise, svd_op):
    """
    Parameters
    ----------
    noise : np.array
        2-D array of shape num_samples x ?
    operator: svd_op
        svd_op.V has to be of shape ? x ?

    Returns
    -------
    np.array
    
    The coefficients Delta_n that correspond to the noise and the singular vectors svd_op.V
    """
    num_samples = noise.shape[-1]
    return np.sum((svd_op.V.T@noise)**2, axis = 1)/num_samples
