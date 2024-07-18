import numpy as np


def getEssentialMatrix(K, F):
    E = K.T @ F @ K
    U, s, V = np.linalg.svd(E)
    s = [1, 1, 0]
    E_ = U @ np.diag(s) @ V
    return E_
