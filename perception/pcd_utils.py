import numpy as np
import open3d as o3d
import sys
import cv2

def resize(img):
    """resizing for calibration / UOIS input preprocess"""
    return cv2.resize(img[:,172:1772], dsize=(640,480), interpolation=cv2.INTER_AREA)

def shrink_mask(obj_mask, filter_size=15, threshold=170):
    """threshold: higher=smaller"""
    rgb_mask = cv2.cvtColor(np.float32(obj_mask*256), cv2.COLOR_GRAY2RGB)
    blurred = cv2.blur(rgb_mask, (filter_size, filter_size))
    shrinked_mask = np.where(blurred[:,:,0] >= threshold, 1, 0)
    return shrinked_mask

def remove_outlier(pcd_o3d, nb_neighbors, std_ratio):
    _, inlier_index = pcd_o3d.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                            std_ratio=std_ratio)
    return pcd_o3d.select_by_index(inlier_index)

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def to_pointcloud(xyz):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))

def get_removal_index_outside_workspace(xyz, 
        x_thresholds=[-0.4, 0.2], 
        y_thresholds=[-0.4, 0.4],
        z_thresholds=[-0.02, np.inf]):
    inside_x = (x_thresholds[0] < xyz[:,0]) & (xyz[:,0] < x_thresholds[1])
    inside_y = (y_thresholds[0] < xyz[:,1]) & (xyz[:,1] < y_thresholds[1])
    inside_z = (z_thresholds[0] < xyz[:,2]) & (xyz[:,2] < z_thresholds[1])
    return inside_x&inside_y&inside_z

def get_pcd_with_base(pcd_o3d):
    pcd = np.asarray(pcd_o3d.points)
    pcd_base_2d = pcd[:,:2]
    pcd_base_3d = np.concatenate((pcd_base_2d, np.zeros((pcd_base_2d.shape[0],1))),1)
    pcd_base_3d_down = np.zeros((700, 3))
    pcd_base_3d_down[0] = pcd_base_3d[np.random.randint(len(pcd_base_3d))]
    distances = np.full(pcd_base_3d.shape[0], np.inf)
    for i in range(1, 700):
        distances = np.minimum(distances, np.linalg.norm(pcd_base_3d - pcd_base_3d_down[i - 1], axis=1))
        pcd_base_3d_down[i] = pcd_base_3d[np.argmax(distances)]
    
    pcd_with_base = np.concatenate((pcd, pcd_base_3d_down))
    pcd_with_base_o3d = o3d.geometry.PointCloud()
    pcd_with_base_o3d.points = o3d.utility.Vector3dVector(pcd_with_base)
    return pcd_with_base_o3d

def farthest_point_sampling(points, num_samples):
    points = np.array(points)
    farthest_points = np.zeros((num_samples, 3))
    farthest_points[0] = points[np.random.randint(len(points))]
    distances = np.full(points.shape[0], np.inf)
    for i in range(1, num_samples):
        distances = np.minimum(distances, np.linalg.norm(points - farthest_points[i - 1], axis=1))
        farthest_points[i] = points[np.argmax(distances)]

    farthest_points_o3d = o3d.geometry.PointCloud()
    farthest_points_o3d.points = o3d.utility.Vector3dVector(farthest_points)
    return farthest_points_o3d

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.9999))
    return result_ransac


def refine_registration(source, target, result_ransac, voxel_size): #source_fpfh, target_fpfh, 
    distance_threshold = voxel_size * 0.02
    print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result