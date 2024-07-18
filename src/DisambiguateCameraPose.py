import numpy as np

# Check if the depth of the 3D points is positive
def DepthPositivityConstraint(x3D, r3, C):
    """
    x3D: 3D points
    r3: The third row of the rotation matrix
    C: Camera center
    return: The number of 3D points with positive depth
    """
    # r3(X-C) alone doesnt solve the check positivity. z = X[2] must also be +ve
    n_positive_depths = 0
    for X in x3D:
        X = X.reshape(-1, 1)
        if r3.dot(X-C) > 0 and X[2] > 0:
            n_positive_depths += 1
    return n_positive_depths


# To select the best camera pose from the output of the PnP RANSAC algorithm.
def DisambiguatePose(r_set, c_set, x3D_set):
    """
    r_set: A list of rotation matrices
    c_set: A list of camera centers
    x3D_set: A list of 3D points
    return: The best rotation matrix, camera center, and 3D points
    """

    best_i = 0
    max_positive_depths = 0

    for i in range(len(r_set)):
        R = r_set[i]
        C = c_set[i].reshape(-1, 1)
        r3 = R[2, :].reshape(1, -1)
        x3D = x3D_set[i]
        x3D = x3D / x3D[:, 3].reshape(-1, 1)
        x3D = x3D[:, 0:3]
        n_positive_depths = DepthPositivityConstraint(x3D, r3, C)
        if n_positive_depths > max_positive_depths:
            best_i = i
            max_positive_depths = n_positive_depths

    R, C, x3D = r_set[best_i], c_set[best_i], x3D_set[best_i]
    return R, C, x3D