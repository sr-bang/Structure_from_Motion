import numpy as np

# Extract the camera pose from the essential matrix
def ExtractCameraPose(E):
    """
    E: Essential matrix
    return: A list of rotation matrices and camera centers
    """

    U, S, V_T = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R = []
    C = []
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    C.append(U[:, 2])
    C.append(-U[:, 2])
    C.append(U[:, 2])
    C.append(-U[:, 2])

    for i in range(4):
        if (np.linalg.det(R[i]) < 0): # Ensure the determinant of the rotation matrix is positive
            R[i] = -R[i]
            C[i] = -C[i]

    return R, C
