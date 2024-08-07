import numpy as np
import cv2
import scipy.optimize as optimize

# To calculate the reprojection error for the given 3D point X
def ReprojectionError(X, pt1, pt2, R1, C1, R2, C2, K):
    """
    x: 3D point
    pt1: 2D point in image 1
    pt2: 2D point in image 2
    R1: Rotation matrix of camera 1
    C1: Camera center of camera 1
    R2: Rotation matrix of camera 2
    C2: Camera center of camera 2
    K: Intrinsic matrix of the camera
    return: reprojection error for the given 3D point
    """
    P1 = ProjectionMatrix(R1, C1, K)
    P2 = ProjectionMatrix(R2, C2, K)

    # X = homo(X.reshape(1,-1)).reshape(-1,1) # make X it a column of homogenous vector

    p1_1T, p1_2T, p1_3T = P1  # rows of P1
    p1_1T, p1_2T, p1_3T = p1_1T.reshape(
        1, -1), p1_2T.reshape(1, -1), p1_3T.reshape(1, -1)

    p2_1T, p2_2T, p2_3T = P2  # rows of P2
    p2_1T, p2_2T, p2_3T = p2_1T.reshape(
        1, -1), p2_2T.reshape(1, -1), p2_3T.reshape(1, -1)

    # reprojection error for reference camera points - j = 1
    u1, v1 = pt1[0], pt1[1]
    u1_proj = np.divide(p1_1T.dot(X), p1_3T.dot(X))
    v1_proj = np.divide(p1_2T.dot(X), p1_3T.dot(X))
    E1 = np.square(v1 - v1_proj) + np.square(u1 - u1_proj)

    # reprojection error for second camera points - j = 2
    u2, v2 = pt2[0], pt2[1]
    u2_proj = np.divide(p2_1T.dot(X), p2_3T.dot(X))
    v2_proj = np.divide(p2_2T.dot(X), p2_3T.dot(X))

    E2 = np.square(v2 - v2_proj) + np.square(u2 - u2_proj)

    return E1, E2

# Mean of reprojection error
def meanReprojectionError(x3D, pts1, pts2, R1, C1, R2, C2, K):
    """
    x3D: 3D point
    pts1: 2D point in image 1
    pts2: 2D point in image 2
    R1: Rotation matrix of camera 1
    C1: Camera center of camera 1
    R2: Rotation matrix of camera 2
    C2: Camera center of camera 2
    K: Intrinsic matrix of the camera
    return: mean reprojection error for the given 3D point
    
    """
    Error = []
    for pt1, pt2, X in zip(pts1, pts2, x3D):
        e1, e2 = ReprojectionError(X, pt1, pt2, R1, C1, R2, C2, K)
        Error.append(e1+e2)

    return np.mean(Error)

# To calculate the projection matrix
def ProjectionMatrix(R, C, K):
    """
    R: Rotation matrix
    C: Camera center
    K: Intrinsic matrix
    return: Projection matrix
    """
    C = np.reshape(C, (3, 1))
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))
    return P

# To calculate the reprojection loss
def NonLinearTriangulation(K, pts1, pts2, x3D, R1, C1, R2, C2):
    """
    K: Intrinsic matrix
    pts1: 2D point in image 1
    pts2: 2D point in image 2
    x3D: 3D point
    R1: Rotation matrix of camera 1
    C1: Camera center of camera 1
    R2: Rotation matrix of camera 2
    C2: Camera center of camera 2
    return: 3D points after optimization
    """
    P1 = ProjectionMatrix(R1, C1, K)
    P2 = ProjectionMatrix(R2, C2, K)
    # pts1, pts2, x3D = pts1, pts2, x3D

    if pts1.shape[0] != pts2.shape[0] != x3D.shape[0]:
        raise 'Check point dimensions - level nlt'

    x3D_ = []
    for i in range(len(x3D)):
        optimized_params = optimize.least_squares(
            fun=ReprojectionLoss, x0=x3D[i], method="trf", args=[pts1[i], pts2[i], P1, P2])
        X1 = optimized_params.x
        x3D_.append(X1)
        # x3D_.append(X1[:3])
    return np.array(x3D_)


# To calculate the reprojection loss
def ReprojectionLoss(X, pts1, pts2, P1, P2):
    """
    X: 3D point
    pts1: 2D point in image 1
    pts2: 2D point in image 2
    P1: Projection matrix of camera 1
    P2: Projection matrix of camera 2
    return: reprojection error for the given 3D point
    """
    p1_1T, p1_2T, p1_3T = P1  # rows of P1
    p1_1T, p1_2T, p1_3T = p1_1T.reshape(
        1, -1), p1_2T.reshape(1, -1), p1_3T.reshape(1, -1)

    p2_1T, p2_2T, p2_3T = P2  # rows of P2
    p2_1T, p2_2T, p2_3T = p2_1T.reshape(
        1, -1), p2_2T.reshape(1, -1), p2_3T.reshape(1, -1)

    # reprojection error for reference camera points - j = 1
    u1, v1 = pts1[0], pts1[1]
    u1_proj = np.divide(p1_1T.dot(X), p1_3T.dot(X))
    v1_proj = np.divide(p1_2T.dot(X), p1_3T.dot(X))
    E1 = np.square(v1 - v1_proj) + np.square(u1 - u1_proj)

    # reprojection error for second camera points - j = 2
    u2, v2 = pts2[0], pts2[1]
    u2_proj = np.divide(p2_1T.dot(X), p2_3T.dot(X))
    v2_proj = np.divide(p2_2T.dot(X), p2_3T.dot(X))
    E2 = np.square(v2 - v2_proj) + np.square(u2 - u2_proj)

    error = E1 + E2
    return error.squeeze()

# To make the 3D point homogenous
def homo(pts):
    return np.hstack((pts, np.ones((pts.shape[0], 1))))
