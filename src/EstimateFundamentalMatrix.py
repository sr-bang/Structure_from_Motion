import numpy as np

# Normalize the points before computing the fundamental matrix
def normalize(uv):
    """
    uv: Points
    return: Normalized points and the transformation matrix
    """
    uv_ = np.mean(uv, axis=0)
    u_, v_ = uv_[0], uv_[1]
    u_cap, v_cap = uv[:, 0] - u_, uv[:, 1] - v_

    s = (2/np.mean(u_cap**2 + v_cap**2))**(0.5)
    T_scale = np.diag([s, s, 1])
    T_trans = np.array([[1, 0, -u_], [0, 1, -v_], [0, 0, 1]])
    T = T_scale.dot(T_trans)

    x_ = np.column_stack((uv, np.ones(len(uv))))
    x_norm = (T.dot(x_.T)).T

    return x_norm, T

# Get the fundamental matrix from the points
def get_f_mat(pts1, pts2):
    """
    pts1: Points in the first image
    pts2: Points in the second image
    return: The fundamental matrix
    """
    normalised = True
    x1, x2 = pts1, pts2

    if x1.shape[0] > 7:
        if normalised == True:
            x1_norm, T1 = normalize(x1)
            x2_norm, T2 = normalize(x2)
        else:
            x1_norm, x2_norm = x1, x2

        A = np.zeros((len(x1_norm), 9))
        for i in range(0, len(x1_norm)):
            x_1, y_1 = x1_norm[i][0], x1_norm[i][1]
            x_2, y_2 = x2_norm[i][0], x2_norm[i][1]
            A[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2 *
                            x_1, y_2*y_1, y_2, x_1, y_1, 1])

        U, S, VT = np.linalg.svd(A, full_matrices=True) 
        F = VT.T[:, -1]
        F = F.reshape(3, 3)

        u, s, vt = np.linalg.svd(F)
        s = np.diag(s)
        s[2, 2] = 0
        F = np.dot(u, np.dot(s, vt))

        if normalised:
            F = np.dot(T2.T, np.dot(F, T1))
            F = F / F[2, 2]
        return F

    else:
        return None
