import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd
import cv2
from tqdm import trange
from argparse import Namespace
import os
from showPointCloud import displayPointcloud

cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])    
distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])
iterations = 35

images_df = pd.read_pickle("data/images.pkl")
train_df = pd.read_pickle("data/train.pkl")
points3D_df = pd.read_pickle("data/points3D.pkl")
point_desc_df = pd.read_pickle("data/point_desc.pkl")

def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

def pnpsolver(query,model,cameraMatrix=0,distortion=0):
    kp_query, desc_query = query
    kp_model, desc_model = model

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_query,desc_model,k=2)

    gmatches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            gmatches.append(m)

    points2D = np.empty((0,2))
    points3D = np.empty((0,3))

    for mat in gmatches:
        query_idx = mat.queryIdx
        model_idx = mat.trainIdx
        points2D = np.vstack((points2D,kp_query[query_idx]))
        points3D = np.vstack((points3D,kp_model[model_idx]))

    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])    
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])

    return cv2.solvePnPRansac(points3D, points2D, cameraMatrix, distCoeffs)

def image_undistortion(points, distCoeffs):
    size = [1080, 1920]
    points = (points.T/np.array([size[0], size[1]]).reshape((2, 1)))

    center = np.array([0.5, 0.5]).reshape((2, 1))
    r = np.linalg.norm((points - center), axis=0)

    xc, yc = center[0], center[1]
    x, y = points[0], points[1]
    k1, k2, p1, p2 = distCoeffs[0], distCoeffs[1], distCoeffs[2], distCoeffs[3]

    xu = x + (x - xc) * (k1 * (r**2) + k2 * (r**4)) + \
    (p1 * (r**2 + 2*((x - xc)**2)) + 2 * p2 * (x - xc) * (y - yc))
    yu = y + (y - yc) * (k1 * (r**2) + k2 * (r**4)) + \
    (p2 * (r**2 + 2*((y - yc)**2)) + 2 * p1 * (x - xc) * (y - yc))
    undistorted_points = np.vstack((xu, yu)) * np.array([size[0], size[1]]).reshape((2, 1))

    return undistorted_points.T

def cosineSimilarity(v, w):
    return np.dot(v, w)/(np.linalg.norm(v)* np.linalg.norm(w))

def trilaterate3D(points, distances):
    p1, p2, p3 = points[0], points[1], points[2]
    r1, r2, r3 = distances[0], distances[1], distances[2]

    v_1 = (p2 - p1) / (np.linalg.norm(p2 - p1))
    v_2 = (p3 - p1) / (np.linalg.norm(p3 - p1))

    ix = v_1
    iz = np.cross(v_1, v_2) / np.linalg.norm(np.cross(v_1, v_2))
    iy = np.cross(ix, iz) / np.linalg.norm(np.cross(ix, iz))

    x2 = np.linalg.norm(p2 - p1)
    x3 = (p3-p1)@ix
    y3 = (p3-p1)@iy
    
    x_length = (r1**2 - r2**2 + x2**2)/(2*x2)


    y_length = (r1**2 - r3**2 + x3**2 + y3**2 - (2*x3*x_length))/(2*y3)
    

    z_length = np.sqrt(r1**2 - x_length**2 - y_length**2)

    direction = x_length*ix + y_length*iy + z_length*iz
    direction_minus = x_length*ix + y_length*iy - z_length*iz

    return [np.array(p1+direction), np.array(p1+direction_minus)]

def find_matches(query, model):
    kp_query, desc_query = query
    kp_model, desc_model = model

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_query,desc_model,k=2)

    gmatches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            gmatches.append(m)

    points2D = np.empty((0,2))
    points3D = np.empty((0,3))

    for mat in gmatches:
        query_idx = mat.queryIdx
        model_idx = mat.trainIdx
        points2D = np.vstack((points2D,kp_query[query_idx]))
        points3D = np.vstack((points3D,kp_model[model_idx]))
    
    return points2D, points3D


def p3pSolver(points2D, points3D, cameraMatrix=0):
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])

    points2D = points2D.reshape(4,2)
    points2D_ccs = np.insert(points2D, 2, 1, axis=1).transpose()
    cameraMatrix_inv = np.linalg.inv(cameraMatrix)
    points2D_ccs = np.dot(cameraMatrix_inv, points2D_ccs)
    points2D_ccs = (points2D_ccs / np.linalg.norm(points2D_ccs, axis=0)).transpose()

    x1, x2, x3, x4 = points3D[0], points3D[1], points3D[2], points3D[3] # points in ecs
    u1, u2, u3, u4 = points2D[0], points2D[1], points2D[2], points2D[3] # points in pcs
    v1, v2, v3, v4 = points2D_ccs[0], points2D_ccs[1], points2D_ccs[2], points2D_ccs[3] # points in ccs

    cos_ab = cosineSimilarity(v1, v2)
    cos_ac = cosineSimilarity(v1, v3)
    cos_bc = cosineSimilarity(v2, v3)

    Rab = np.linalg.norm(x1-x2)
    Rac = np.linalg.norm(x1-x3)
    Rbc = np.linalg.norm(x2-x3)

    K1 = (Rbc/Rac)**2
    K2 = (Rbc/Rab)**2
    # print('cos_ab, cos_ac, cos_bc, Rab, Rac, Rbc, K1, K2: \n', cos_ab, cos_ac, cos_bc, Rab, Rac, Rbc, K1, K2,'\n')

    G4 = (K1*K2-K1-K2)**2 - 4*K1*K2*(cos_bc**2)
    G3 = 4*(K1*K2-K1-K2)*K2*(1-K1)*cos_ab + 4*K1*cos_bc*((K1*K2-K1+K2)*cos_ac+2*K2*cos_ab*cos_bc)
    G2 = (2*K2*(1-K1)*cos_ab)**2 + 2*(K1*K2-K1-K2)*(K1*K2+K1-K2) + 4*K1*((K1-K2)*(cos_bc**2)+K1*(1-K2)*(cos_ac**2)-2*(1+K1)*K2*cos_ab*cos_ac*cos_bc)
    G1 = 4*(K1*K2+K1-K2)*K2*(1-K1)*cos_ab + 4*K1*((K1*K2-K1+K2)*cos_ac*cos_bc+2*K1*K2*cos_ab*(cos_ac**2))
    G0 = (K1*K2+K1-K2)**2 - 4*(K1**2)*K2*(cos_ac**2)
    # print('(%f)x^4+(%f)x^3+(%f)x^2+(%f)x+(%f)'%(G4, G3, G2, G1, G0))

    try:
        x = np.roots([G4, G3, G2, G1, G0])
        x = x[np.isreal(x)].real
        # print('x:', x)

        m = 1 - K1
        p = 2*(K1*cos_ac-x*cos_bc)
        q = x**2 - K1
        m_prime = 1
        p_prime = 2*(-x)*cos_bc
        q_prime = x**2*(1-K2) + 2*x*K2*cos_ab - K2

        b1 = p*m_prime - p_prime*m
        b0 = m_prime*q - m*q_prime
        y = -b0 / b1

        a = np.sqrt((Rab ** 2) / (1 + (x ** 2) - 2 * x * cos_ab))
        b = x * a
        c = y * a
        # print('x, y, a, b, c:',x, y ,a, b, c)

        Ts = []
        for i in range(len(a)):
            T1, T2 = trilaterate3D([x1, x2, x3], [a[i], b[i], c[i]])
            Ts.append(T1)
            Ts.append(T2)


        min_error = np.Inf
        model = []
        for T in Ts:
            for sign in [1, -1]:
                lambda_ = sign * np.linalg.norm((points3D[:3] - T), axis=1)
                R_mat = (lambda_ * points2D_ccs[:3].T) @ np.linalg.pinv((points3D[:3]-T).T)

                v4_hat = cameraMatrix @ R_mat @ (x4-T)
                v4_hat = v4_hat/v4_hat[-1]

                if min_error > np.linalg.norm(v4 - v4_hat):
                    min_error = np.linalg.norm(v4 - v4_hat)
                    model = [R_mat, T]

        # print('R_mat, T_vec: ', model[0], model[1])
        rotq = R.from_matrix(model[0]).as_quat()
        return model[0], model[1], min_error
    except:
        print('No solution for these three points')
        return None

def ransac(points2D, points3D, cameraMatrix, distCoeffs):
    max_inliers = -1
    best_model = []
    original_total = points3D.shape[0]

    param = Namespace(
            s = 4,
            e = 0.5,
            p = 0.99,
            d = 10,
        )

    num_iter = int(np.ceil((np.log(1-param.p))/(np.log(1-(1-param.e)**param.s))))
    undistortPoints2D = points2D
    # undistortPoints2D = cv2.undistortPoints(points2D, cameraMatrix = cameraMatrix, distCoeffs = distCoeffs)

    for i in trange(num_iter):
        ids = np.random.choice([i for i in range(len(points2D))], 4)
        print('id: ', ids)

        points3D_choosen = np.array([points3D[ids[0]], points3D[ids[1]], points3D[ids[2]], points3D[ids[3]]]) # points in ecs
        points2D_choosen = np.array([undistortPoints2D[ids[0]], undistortPoints2D[ids[1]], undistortPoints2D[ids[2]], undistortPoints2D[ids[3]]]) # points in pcs

        try:
            R_mat, T_vec, error = p3pSolver(points2D_choosen, points3D_choosen, cameraMatrix)
            print(R_mat)
            projected2D = cameraMatrix @ (R_mat @ (points3D - T_vec).T)
            projected2D = (projected2D / projected2D[2])
            projected2D = projected2D[:2].T

            error = np.linalg.norm((points2D - projected2D), axis = 1)
            #print(error)

            inliers = np.where(error < param.d)[0].shape[0]
            inlier_id = np.where(error < param.d)[0]

            if (inliers >= int(original_total*param.e)):
                print('\nUpdating Model.....')
                # print('Number of Inliers: ', inliers)
                max_inliers = inliers
                points3D = points3D[inlier_id]
                points2D = points2D[inlier_id]
                best_model = [R_mat, T_vec]

        except:
            print('No Solution')
            pass

    print('\n\n max inliers: ', max_inliers)
    print('model', best_model)
    return best_model

def ransacP3P(query, model, cameraMatrix, distortion):
  print('\nFinding Matches...')
  points2D, points3D = find_matches(query, model)
  print('Strat Ransac')
  best_model = ransac(points2D, points3D, cameraMatrix, distortion)

  return best_model

def calculate_error(Rs, Ts, gt_Rs, gt_Ts):
  error_Rs = []
  error_Ts = []
  for i in range(len(Rs)):
      error_Ts.append(np.linalg.norm(Ts[i] - gt_Ts[i]))
      norm_R = Rs[i] / np.linalg.norm(Rs[i])
      norm_gt_R = gt_Rs[i] / np.linalg.norm(gt_Rs[i])
      diff_R = np.clip(np.sum(norm_R * norm_gt_R), 0, 1)
      error_Rs.append(np.degrees(np.arccos(2 * diff_R * diff_R - 1)))        
  error_Rs = np.array(error_Rs)
  error_Ts = np.array(error_Ts)

  return np.median(error_Rs), np.median(error_Ts)

def main(cameraMatrix, distCoeffs):
    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)

    if os.path.isfile("undistortCameraPose.pkl"):
        print('Already have pose information')
        poses_df = pd.DataFrame(pd.read_pickle("undistortCameraPose.pkl")) # {'R_pred': R_pred, 'T_pred': T_pred, 'R_gt': R_gt, 'T_gt': T_gt}
        R_pred = []
        T_pred = []
        for i in range(len(poses_df['R_pred'])):
            R_pred.append(poses_df['R_pred'][i])
            T_pred.append(poses_df['T_pred'][i])
        R_pred = np.array(R_pred)
        T_pred = np.array(T_pred)
    else:
        img_ids = images_df["IMAGE_ID"].unique().tolist()
        R_pred = []
        T_pred = []
        R_gt = []
        T_gt = []
        for i in trange(len(img_ids)):
            idx = img_ids[i]

            # Load quaery image
            fname = ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]
            rimg = cv2.imread("data/frames/"+fname,cv2.IMREAD_GRAYSCALE)

            # Load query keypoints and descriptors
            points = point_desc_df.loc[point_desc_df["IMAGE_ID"]==idx]
            kp_query = np.array(points["XY"].to_list())
            desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

            #Ransac P#P
            [R_mat, T_vec] = ransacP3P((kp_query, desc_query),(kp_model, desc_model), cameraMatrix, distCoeffs)

            rotq_hat = R.from_matrix(R_mat).as_rotvec() # predict rotation
            tvec_hat = T_vec # predict translation
            R_pred.append(rotq_hat)
            T_pred.append(tvec_hat)

            # Get camera pose groudtruth 
            ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx]
            rotq_gt = R.from_quat(ground_truth[["QX","QY","QZ","QW"]].values).as_rotvec()[0]
            tvec_gt = ground_truth[["TX","TY","TZ"]].values
            rotq_gt_mat = R.from_quat(ground_truth[["QX","QY","QZ","QW"]].values).as_matrix()
            tvec_gt_actually = np.dot(-1*np.linalg.inv(rotq_gt_mat), tvec_gt.T) # translation of groundtruth is actually -inv(R)@T
            tvec_gt_actually = tvec_gt_actually.reshape(1,3)[0]
            R_gt.append(rotq_gt.tolist())
            T_gt.append(tvec_gt_actually.tolist())

        print('Saving Camera Pose......')
        RT_dict = {'R_pred': R_pred, 'T_pred': T_pred, 'R_gt': R_gt, 'T_gt': T_gt}
        with open('undistortCameraPose.pkl', 'wb') as f:
            pickle.dump(RT_dict, f)
        print('Saving Camera Pose Finish\n')

        R_pred = np.array(R_pred)
        T_pred = np.array(T_pred)
        R_gt = np.array(R_gt)
        T_gt = np.array(T_gt)
        # print('R_pred: \n', R_pred)
        # print('T_pred: \n',T_pred)
        # print('R_gt: \n',R_gt)
        # print('T_gt: \n',T_gt)
        error_R, error_T = calculate_error(R_pred, T_pred, R_gt, T_gt)
        print('error_R: %f, error_T: %f'%(error_R, error_T))

    displayPointcloud((kp_model, np.array(desc_df['RGB'].to_list())), (R_pred, T_pred))

if __name__ == '__main__':
    main(cameraMatrix, distCoeffs)