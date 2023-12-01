import numpy as np
from EstimateFundamentalMatrix import get_f_mat

def errorF(pts1, pts2, F):

    # Checking the epipolar constraint
    x1,x2 = pts1, pts2
    x1tmp=np.array([x1[0], x1[1], 1])
    x2tmp=np.array([x2[0], x2[1], 1]).T

    error = np.dot(x2tmp, np.dot(F, x1tmp))

    return np.abs(error)

def ransac(pts1, pts2, idx):
    n_iterations = 2000
    error_thresh = 0.002
    inliers_thresh = 0
    chosen_indices = []
    chosen_f = None

    for _ in range(n_iterations):
        
        n_rows = pts1.shape[0] 
        random_indices = np.random.choice(n_rows, size=8) #selecting 8 points randomly
        pts1_8 = pts1[random_indices, :] 
        pts2_8 = pts2[random_indices, :] 
        f_8 = get_f_mat(pts1_8, pts2_8)
        indices = []

        if f_8 is not None:
            for j in range(n_rows):

                error = errorF(pts1[j, :], pts2[j, :], f_8)

                if error < error_thresh:
                    indices.append(idx[j])

        if len(indices) > inliers_thresh:
            inliers_thresh = len(indices)
            chosen_indices = indices
            chosen_f = f_8

    return chosen_f, chosen_indices