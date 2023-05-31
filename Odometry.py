from glob import glob
import cv2, skimage, os
import numpy as np

class OdometryClass:
    def __init__(self, frame_path):
        self.frame_path = frame_path
        self.frames = sorted(glob(os.path.join(self.frame_path, 'images', '*')))
        with open(os.path.join(frame_path, 'calib.txt'), 'r') as f:
            lines = f.readlines()
            self.focal_length = float(lines[0].strip().split()[-1])
            lines[1] = lines[1].strip().split()
            self.pp = (float(lines[1][1]), float(lines[1][2]))
            
        with open(os.path.join(self.frame_path, 'gt_sequence.txt'), 'r') as f:
            self.pose = [line.strip().split() for line in f.readlines()]

    def imread(self, fname):
        """
        read image into np array from file
        """
        return cv2.imread(fname, 0)

    def imshow(self, img):
        """
        show image
        """
        skimage.io.imshow(img)

    def get_gt(self, frame_id):
        pose = self.pose[frame_id]
        x, y, z = float(pose[3]), float(pose[7]), float(pose[11])
        return np.array([[x], [y], [z]])

    def get_scale(self, frame_id):
        '''Provides scale estimation for mutliplying
        translation vectors
        
        Returns:
        float -- Scalar value allowing for scale estimation
        '''
        prev_coords = self.get_gt(frame_id - 1)
        curr_coords = self.get_gt(frame_id)
        return np.linalg.norm(curr_coords - prev_coords)

    def get_calib_matrix(self):
        r1 = np.array([self.focal_length, 0, self.pp[0]])
        r2 = np.array([0, self.focal_length, self.pp[1]])
        r3 = np.array([0, 0, 1])
        K = np.vstack([r1,r2,r3])
        return K
    
    def rt_transform(self, R, t):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
        
    def get_matches(self, img1_fname, img2_fname):
        img1 = self.imread(img1_fname)
        img2 = self.imread(img2_fname)

        orb = cv2.ORB_create(3000)

        kpts1, kpts1_des = orb.detectAndCompute(img1, None)
        kpts2, kpts2_des = orb.detectAndCompute(img2, None)

        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        matches = flann.knnMatch(kpts1_des, kpts2_des, k=2)

        good_matches = []
        try:
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)
        except ValueError:
            pass

        p1, p2 = [], [] 
        for p in good_matches:
            p1.append(kpts1[p.queryIdx].pt)
            p2.append(kpts2[p.trainIdx].pt)

        return np.float32(p1), np.float32(p2)
    
    def normalized_8p_alg(self, p1, p2):
        p_num = p1.shape[0]

        p1_mean = np.mean(p1, axis=0)
        p2_mean = np.mean(p2, axis=0)

        # center points
        p1_center = p1 - p1_mean
        p2_center = p2 - p2_mean

        # get scale
        s1 = np.sqrt(2/(np.sum(np.square(p1_center))/p_num))
        s2 = np.sqrt(2/(np.sum(np.square(p2_center))/p_num))

        T1_1 = np.array([s1, 0, p1_mean[0] * -s1])
        T1_2 = np.array([0, s1, p1_mean[1] * -s1])

        T1 = np.vstack((T1_1, T1_2, np.array([0,0,1])))

        T2_1 = np.array([s2, 0, p2_mean[0] * -s2])
        T2_2 = np.array([0, s2, p2_mean[1] * -s2])

        T2 = np.vstack((T2_1, T2_2, np.array([0,0,1])))

        p1_ones = np.hstack((p1, np.ones((p_num, 1))))
        p2_ones = np.hstack((p2, np.ones((p_num, 1))))

        p1_norm = np.dot(T1, p1_ones.T).T
        p2_norm = np.dot(T2, p2_ones.T).T

        # use 8-point alg to calculate F with normalized points
        F_norm = self.eight_point_alg(p1_norm, p2_norm)
        
        # de-normalize F
        F = np.dot(np.dot(T2.T, F_norm), T1)

        return F 
    
    def eight_point_alg(self, p1, p2):
        p_num = p1.shape[0]

        A = np.zeros((p_num, 9))

        for i in range(p_num):
            x1 = p1[i][0]
            y1 = p1[i][1]
            x2 = p2[i][0]
            y2 = p2[i][1]
            A[i] = np.array([x1*x2, x2*y1, x2, y2*x1, y1*y2, y2, x1, y1, 1])
           
        _, _, V0 = np.linalg.svd(A)

        f = V0[-1:,:]

        f = np.reshape(f, (3,3))

        # enforce rank 2
        U, S, V1 = np.linalg.svd(f)

        s = np.zeros((3,3))
        s[0][0] = S[0]
        s[1][1] = S[1]

        F = np.dot(np.dot(U,s), V1)

        return F / F[-1, -1] 
    
    def get_F_ransac(self, p1, p2, threshold = 0.1):
        p_num = p1.shape[0]
        most_inliers= 0
        best_F = []
        max_iters = 750
        curr_inliers = 0
        for _ in range(max_iters):
            idx = np.random.randint(p1.shape[0], size = 8)

            rp1, rp2 = p1[idx,: ], p2[idx,: ]

            F_curr = self.normalized_8p_alg(np.array(rp1), np.array(rp2))

            pts1 = np.hstack((p1, np.ones((p_num,1))))
            pts2 = np.hstack((p2, np.ones((p_num,1))))
            
            err1 =  np.dot(pts1, F_curr.T)
            err2 =  np.dot(pts2, F_curr)

            errors = np.square(np.sum(err2 * pts1, axis = 1)) / np.sum(np.square(np.hstack((err1[:,:-1],err2[:,:-1]))), axis = 1)
            
            inliers = [errors <= threshold]
            curr_inliers = np.sum(inliers)
            if most_inliers < curr_inliers:
                most_inliers = curr_inliers
                best_F = F_curr
            
        return best_F


    def get_Rt(self, E_M, p1, p2, K):
        _, R, t, _ = cv2.recoverPose(E_M, p1, p2, K)
        return R, t
    
    def get_T_matrix(self, p1, p2, K):
        F = self.get_F_ransac(p1, p2)
        
        E_M = np.dot(np.dot(K.T, F), K)
        
        R,t = self.get_Rt(E_M, p1, p2, K)

        T_M = self.rt_transform(R, t.flatten())

        return T_M

    def run(self):
        """
        Uses the video frame to predict the path taken by the camera
        
        The reurned path should be a numpy array of shape nx3
        n is the number of frames in the video  
        """
        path = []

        K = self.get_calib_matrix()
        
        for i in range(len(self.frames)-1):
            if i == 0:
                pose = np.hstack((np.eye(3), np.array([[0,0,0]]).T))
            else:
                p1,p2 = self.get_matches(self.frames[i-1], self.frames[i])
                
                T_M = self.get_T_matrix(p1,p2,K)

                pose = np.dot(pose, np.linalg.inv(T_M))

            pred = pose.flatten()
            path.append([pred[3], pred[7], pred[11]])

        path = np.array(path)
        np.save('predictions', path)
        return path

        
if __name__=="__main__":
    frame_path = 'video_train'
    odemotryc = OdometryClass(frame_path)
    path = odemotryc.run()
    print(path,path.shape)
