'''Slightly Inspired from https://github.com/sakshikakde/SFM'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
from glob import glob
import os
from GetInliersRANSAC import ransac
from Visualize import *
from EssentialMatrixFromFundamentalMatrix import getEssentialMatrix
from ExtractCameraPose import ExtractCameraPose
from LinearTriangulation import LinearTriangulation
from DisambiguateCameraPose import DisambiguatePose
from NonlinearTriangulation import *
from NonlinearPnP import *
from PnPRANSAC import PnPRANSAC
from BundleAdjustment import *

def get_int_mat(path):

    K = []

    with open(path) as file:
        text = csv.reader(file, delimiter=" ")
        for line in text:
            K.append(line)

    K = np.array(K, np.float_)
    return K

def read_imgs(path):

    imgs = []

    for file in glob(path + '*.png'):
        img = cv2.imread(file)
        imgs.append(img)

    # cv2.imshow('img', imgs[2])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return imgs

def get_matches(path):

    feature_descriptor = []
    feature_x = []
    feature_y = []
    feature_flag = []
    paths = glob(path + 'matching*.txt')
    img_count = len(paths) + 1

    for n, path in enumerate(paths):
        with open(path) as file:
            text = csv.reader(file, delimiter=" ")
            next(text)
            for line in text:
                x_row = np.zeros((1, img_count))
                y_row = np.zeros((1, img_count))
                flag_row = np.full((1, img_count), False)

                line = np.array(line[:-1], np.float_)
                
                n_matches = int(line[0])
                r = line[1]
                g = line[2]
                b = line[3]

                feature_descriptor.append([r,g,b])

                src_x = line[4]
                src_y = line[5]

                x_row[0, n] = src_x
                y_row[0, n] = src_y
                flag_row[0, n] = 1

                m = 1
                while n_matches > 1:
                    image_id = int(line[5+m])
                    image_id_x = line[6+m]
                    image_id_y = line[7+m]
                    m += 3
                    n_matches = n_matches - 1

                    x_row[0, image_id - 1] = image_id_x
                    y_row[0, image_id - 1] = image_id_y
                    flag_row[0, image_id - 1] = True

                feature_x.append(x_row)
                feature_y.append(y_row)
                feature_flag.append(flag_row)

    return np.array(feature_x).reshape(-1, img_count), np.array(feature_y).reshape(-1, img_count), np.array(feature_flag).reshape(-1, img_count), np.array(feature_descriptor).reshape(-1, 3)

def main():
    
    data_path = "./Data/"
    calib_path = "calibration.txt"
    img_count = 5
    BA = True

    K = get_int_mat(data_path + calib_path)
    imgs = read_imgs(data_path)
    x, y, flag, desc = get_matches(data_path)

    filt_flag = np.zeros_like(flag)
    f_mat = np.empty((img_count, img_count), dtype=object)

    for i in range(0, img_count - 1):
        for j in range(i + 1, img_count):

            idx = np.where(flag[:,i] & flag[:,j])
            pts1 = np.hstack((x[idx, i].reshape((-1, 1)), y[idx, i].reshape((-1, 1))))
            pts2 = np.hstack((x[idx, j].reshape((-1, 1)), y[idx, j].reshape((-1, 1))))
            idx = np.array(idx).reshape(-1)
            # show_matches(imgs[i], imgs[j], pts1, pts2, (0,255,255), None)
            if len(idx) > 8:
                F_best, chosen_idx = ransac(pts1, pts2, idx)
                print('At image : ', i, j, '|| Number of inliers: ', len(chosen_idx), '/', len(idx))
                f_mat[i, j] = F_best
                filt_flag[chosen_idx, j] = True
                filt_flag[chosen_idx, i] = True
                idx = np.where(filt_flag[:,i] & filt_flag[:,j])
                pts1 = np.hstack((x[idx, i].reshape((-1, 1)), y[idx, i].reshape((-1, 1))))
                pts2 = np.hstack((x[idx, j].reshape((-1, 1)), y[idx, j].reshape((-1, 1))))
                idx = np.array(idx).reshape(-1)
                show_matches(imgs[i], imgs[j], pts1, pts2, (0,255,255), None)

    n, m = 0, 1
    F12 = f_mat[n,m]
    E12 = getEssentialMatrix(K, F12)

    R_set, C_set = ExtractCameraPose(E12)

    idx = np.where(filt_flag[:,n] & filt_flag[:,m])
    pts1 = np.hstack((x[idx, n].reshape((-1, 1)), y[idx, n].reshape((-1, 1))))
    pts2 = np.hstack((x[idx, m].reshape((-1, 1)), y[idx, m].reshape((-1, 1))))

    R1_ = np.identity(3)
    C1_ = np.zeros((3,1))

    pts3D_4 = []
    for i in range(len(C_set)):
        x1 = pts1
        x2 = pts2
        X = LinearTriangulation(K, C1_, R1_, C_set[i], R_set[i], x1, x2)
        X = X/X[:,3].reshape(-1,1)
        pts3D_4.append(X)

    R_chosen, C_chosen, X = DisambiguatePose(R_set, C_set, pts3D_4)
    X = X/X[:,3].reshape(-1,1)
    
    plt.figure("camera disambiguation")
    colors = ['maroon','olive','darkcyan','cornflowerblue']
    for color, X_c in zip(colors, pts3D_4):
        plt.scatter(X_c[:,0],X_c[:,2],color=color,marker='.')

    plt.show()

    X_refined = NonLinearTriangulation(K, pts1, pts2, X, R1_, C1_, R_chosen, C_chosen)
    X_refined = X_refined / X_refined[:,3].reshape(-1,1)
    
    mean_error1 = meanReprojectionError(X, pts1, pts2, R1_, C1_, R_chosen, C_chosen, K )
    mean_error2 = meanReprojectionError(X_refined, pts1, pts2, R1_, C1_, R_chosen, C_chosen, K )

    X_all = np.zeros((x.shape[0], 3))
    camera_indices = np.zeros((x.shape[0], 1), dtype = int) 
    X_found = np.zeros((x.shape[0], 1), dtype = int)

    X_all[idx] = X[:, :3]
    X_found[idx] = 1
    camera_indices[idx] = 1

    X_found[np.where(X_all[:,2] < 0)] = 0

    C_set_ = []
    R_set_ = []

    C0 = np.zeros(3)
    R0 = np.identity(3)
    C_set_.append(C0)
    R_set_.append(R0)

    C_set_.append(C_chosen)
    R_set_.append(R_chosen)

### Registering other cameras

    for i in range(2, img_count):

        print('Registering Image: ', str(i+1) ,'......')
        feature_idx_i = np.where(X_found[:, 0] & filt_flag[:, i])
        if len(feature_idx_i[0]) < 8:
            continue

        pts_i = np.hstack((x[feature_idx_i, i].reshape(-1,1), y[feature_idx_i, i].reshape(-1,1)))
        X = X_all[feature_idx_i, :].reshape(-1,3)
        
        R_init, C_init = PnPRANSAC(K, pts_i, X, n_iterations = 1000, error_thresh = 15)
        errorLinearPnP = reprojectionErrorPnP(X, pts_i, K, R_init, C_init)
        
        Ri, Ci = NonLinearPnP(K, pts_i, X, R_init, C_init)
        errorNonLinearPnP = reprojectionErrorPnP(X, pts_i, K, Ri, Ci)
        print("Error linear PnP: ", errorLinearPnP, " Error non linear PnP: ", errorNonLinearPnP)

        C_set_.append(Ci)
        R_set_.append(Ri)

        for j in range(0, i):
            idx_X_pts = np.where(filt_flag[:, j] & filt_flag[:, i])
            if (len(idx_X_pts[0]) < 8):
                continue

            x1 = np.hstack((x[idx_X_pts, j].reshape((-1, 1)), y[idx_X_pts, j].reshape((-1, 1))))
            x2 = np.hstack((x[idx_X_pts, i].reshape((-1, 1)), y[idx_X_pts, i].reshape((-1, 1))))

            X = LinearTriangulation(K, C_set_[j], R_set_[j], Ci, Ri, x1, x2)
            X = X/X[:,3].reshape(-1,1)
            
            LT_error = meanReprojectionError(X, x1, x2, R_set_[j], C_set_[j], Ri, Ci, K)
            
            X = NonLinearTriangulation(K, x1, x2, X, R_set_[j], C_set_[j], Ri, Ci)
            X = X/X[:,3].reshape(-1,1)
            
            nLT_error = meanReprojectionError(X, x1, x2, R_set_[j], C_set_[j], Ri, Ci, K)
            print("Error LT: ", LT_error, " Error nLT: ", nLT_error)
            
            X_all[idx_X_pts] = X[:,:3]
            X_found[idx_X_pts] = 1
            

        if BA:
            R_set_, C_set_, X_all = BundleAdjustment(X_all,X_found, x, y,
                                                     filt_flag, R_set_, C_set_, K, nCam = i)
           
            for k in range(0, i+1):
                idx_X_pts = np.where(X_found[:,0] & filt_flag[:, k])
                # a = x[idx_X_pts, k].reshape((-1, 1))
                # b = y[idx_X_pts, k].reshape((-1, 1))
                # print(x.shape, X_found.shape, filt_flag.shape)
                # print(idx_X_pts)
                new_x = np.hstack((x[idx_X_pts, k].reshape((-1, 1)), y[idx_X_pts, k].reshape((-1, 1))))
                X = X_all[idx_X_pts]
                BA_error = reprojectionErrorPnP(X, new_x, K, R_set_[k], C_set_[k])
                print("Error BA:", BA_error)
        
    X_found[X_all[:,2]<0] = 0    
    
    feature_idx = np.where(X_found[:, 0])
    X = X_all[feature_idx]
    x = X[:,0]
    y = X[:,1]
    z = X[:,2]
    
    fig = plt.figure(figsize = (10, 10))
    plt.xlim(-250,  250)
    plt.ylim(-100,  500)
    plt.scatter(x, z, marker='.',linewidths=0.5, color = 'blue')
    # for i in range(0, len(C_set_)):
    #     R1 = getEuler(R_set_[i])
    #     R1 = np.rad2deg(R1)
    #     plt.plot(C_set_[i][0],C_set_[i][2], marker=(3, 0, int(R1[1])), markersize=15, linestyle='None')
        
    # plt.savefig(savepath+'2D.png')
    plt.show()
if __name__ == "__main__":
    main()