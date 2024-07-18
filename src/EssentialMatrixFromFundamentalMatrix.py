import numpy as np

# Get the essential matrix from the fundamental matrix
def getEssentialMatrix(K, F):
    """
    K: Camera matrix
    F: Fundamental matrix
    return: Essential matrix
    """
    E = K.T @ F @ K
    U, s, V = np.linalg.svd(E)
    s = [1, 1, 0]
    E_ = U @ np.diag(s) @ V
    return E_
