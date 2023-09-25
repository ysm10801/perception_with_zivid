import os
import numpy as np
from typing import *
import open3d as o3d
import copy
import sys
from pathlib import Path
import trimesh

from .cam import *
from .pcd_utils import *


# source from UOIS
import perception.src.data_augmentation as data_augmentation
import perception.src.segmentation as segmentation
import perception.src.util.utilities as util_


os.environ['CUDA_VISIBLE_DEVICES'] = "0"

class Perception:
    def __init__(self):
        app = zivid.Application()
        zivid_cam_clients = {cam.info.serial_number:cam for cam in app.cameras()}
        zivid_settings = zivid.Settings.load("./Zivid2_Settings_Normal.yml")

        cam_serials = ['21527EB4', '2152CFB3', "f1231819"]

        self.cams: List[Camera] = []
        for idx, serial in enumerate(cam_serials):
            if serial != "f1231819": #zivid
                cam = Zivid(idx, zivid_cam_clients[serial], zivid_settings)
            else: #L515
                cam = L515(idx, serial)
            self.cams.append(cam)

    def calibrate_from_file(self, path):
        path = Path(path)
        for i, cam in enumerate(self.cams):
            cam.T_calib = np.load(path/f"cam{i}.npy")

    def calibration(self, force=False, save=True):
        i = 1
        for cam in self.cams:
            cam.calibrate(force=force, save=save)
            print("Camera ", i, " Calibration Done")
            i+=1
    
    def get_workspace_pcd(self, pcd, cam:Camera):
        pcd_flat = pcd.reshape(-1,3)

        bar = np.ones((pcd_flat.shape[0], 1), dtype=np.float32)
        pcd_temp = np.concatenate((pcd_flat,bar),1)
        pcd_world = np.rot90(np.dot(cam.T_calib, np.rot90(pcd_temp,3)), 1)[:,0:3]

        is_in_workspace = get_removal_index_outside_workspace(pcd_world,
                                                      z_thresholds=[-0.05, 0.2])
        is_not_zero = pcd_flat.sum(axis=-1) != 0

        pcd_o3d = to_pointcloud(pcd_flat[is_not_zero & is_in_workspace])

        _, inlier_index = pcd_o3d.remove_statistical_outlier(
            nb_neighbors=20,std_ratio=0.1)
        indices = np.arange(pcd_flat.shape[0])
        inlier = indices[is_not_zero & is_in_workspace][inlier_index]

        pcd_new = np.zeros((480*640, 3))
        pcd_new[inlier] = pcd_flat[inlier]
        pcd_new = pcd_new.reshape((480, 640, 3))
        return pcd_new

    def set_config(self):
        self.dsn_config = {
            # Sizes
            'feature_dim' : 64, # 32 would be normal

            # Mean Shift parameters (for 3D voting)
            'max_GMS_iters' : 10, 
            'epsilon' : 0.05, # Connected Components parameter
            'sigma' : 0.02, # Gaussian bandwidth parameter
            'num_seeds' : 200, # Used for MeanShift, but not BlurringMeanShift
            'subsample_factor' : 5,
            
            # Misc
            'min_pixels_thresh' : 500,
            'tau' : 15.,
        }
        self.rrn_config = {
            # Sizes
            'feature_dim' : 64, # 32 would be normal
            'img_H' : 224,
            'img_W' : 224,

            # architecture parameters
            'use_coordconv' : False,
        }
        self.uois3d_config = {
            # Padding for RGB Refinement Network
            'padding_percentage' : 0.25,
            
            # Open/Close Morphology for IMP (Initial Mask Processing) module
            'use_open_close_morphology' : True,
            'open_close_morphology_ksize' : 9,
            
            # Largest Connected Component for IMP module
            'use_largest_connected_component' : True,
        }
        checkpoint_dir = '/home/irsl/perception/perception/checkpoints/' # TODO: change this to directory of downloaded models
        self.dsn_filename = checkpoint_dir + 'DepthSeedingNetwork_3D_TOD_checkpoint.pth'
        self.rrn_filename = checkpoint_dir + 'RRN_OID_checkpoint.pth'
        self.uois3d_config['final_close_morphology'] = 'TableTop_v5' in self.rrn_filename

    def get_segmented_mask(self, color:np.ndarray, pcd:np.ndarray, scale=1):
        """
        wrt the camera frame
        Segmentation using UOIS Model
        """
        rgb_imgs = np.zeros((1, 480, 640, 3), dtype=np.float32)
        xyz_imgs = np.zeros((1, 480, 640, 3), dtype=np.float32)

        # RGB
        rgb_img = color
        rgb_imgs[0] = data_augmentation.standardize_image(rgb_img)

        # XYZ
        xyz_imgs[0] = pcd*scale

        batch = {
            'rgb' : data_augmentation.array_to_tensor(rgb_imgs),
            'xyz' : data_augmentation.array_to_tensor(xyz_imgs),
        }

        ### Compute segmentation masks ###
        self.set_config()
        uois_net_3d = segmentation.UOISNet3D(
            self.uois3d_config, 
            self.dsn_filename, 
            self.dsn_config, 
            self.rrn_filename, 
            self.rrn_config)
        fg_masks, center_offsets, initial_masks, seg_masks = uois_net_3d.run_on_batch(batch)

        # Get results in numpy
        seg_masks = seg_masks.cpu().numpy()
        fg_masks = fg_masks.cpu().numpy()
        center_offsets = center_offsets.cpu().numpy().transpose(0,2,3,1)
        initial_masks = initial_masks.cpu().numpy()

        rgb_imgs = util_.torch_to_numpy(batch['rgb'].cpu(), is_standardized_image=True)
        num_objs = np.unique(seg_masks[0,...]).max() + 1
        
        rgb = rgb_imgs[0].astype(np.uint8)
        depth = xyz_imgs[0,...,2]
        seg_mask_plot = util_.get_color_mask(seg_masks[0,...], nc=num_objs)
        
        images = [rgb, depth, seg_mask_plot]
        titles = [f'Image {1}', 'Depth',
                f"Refined Masks. #objects: {np.unique(seg_masks[0,...]).shape[0]-1}"
                ]
        temp_mask = seg_mask_plot[:,:,0]+seg_mask_plot[:,:,1]+seg_mask_plot[:,:,2]
        obj_mask = np.where(temp_mask==0, 0, 1)

        plot_information = (images, titles)
        return obj_mask, plot_information
    
    def get_pcd_wrt_cam(self, cam:Camera, pose_world_cam, scale=1):
        color, pcd = cam.get_rgb_pcd()
        obj_mask_raw, plot_information = self.get_segmented_mask(color, pcd, scale)
        obj_mask = shrink_mask(obj_mask_raw)
        obj_xyz = np.zeros((480, 640, 3), dtype=np.float32)
        for i in range(3):
            obj_xyz[:,:,i] = pcd[:,:,i]*obj_mask
        pcd_cam = obj_xyz.reshape(-1,3)
        pcd_cam_obj_a = []
        for k in range(pcd_cam.shape[0]):
            if np.sum(pcd_cam[k]) != 0:
                pcd_cam_obj_a.append(pcd_cam[k])
        pcd_cam_obj = np.array(pcd_cam_obj_a)
        bar = np.ones((pcd_cam_obj.shape[0], 1), dtype=np.float32)
        pcd_temp = np.concatenate((pcd_cam_obj,bar),1)
        pcd_world = np.rot90(np.dot(pose_world_cam, np.rot90(pcd_temp,3)), 1)[:,0:3]
        return pcd_world, plot_information
    
    def get_object_pcds(self, scale=1):
        pcds = []
        plot_informations = []
        tuning_trans = np.load("./config/tuning_trans_config.npy")
        tuning_rotmtx = np.load("./config/tuning_rotmtx_config.npy")
        for i, cam in enumerate(self.cams):
            pcd, plot_information = self.get_pcd_wrt_cam(cam, cam.T_calib, scale)

            align_tuning = np.r_[np.concatenate((tuning_rotmtx[i], tuning_trans[i]),1),
                                 [[0,0,0,1]]]
            bar = np.ones((pcd.shape[0], 1), dtype=np.float32)
            pcd_temp = np.concatenate((pcd ,bar),1)
            pcd_world = np.rot90(np.dot(align_tuning, np.rot90(pcd_temp,3)), 1)[:,0:3]
            pcds.append(pcd_world)
            plot_informations.append(plot_information)
        return pcds, plot_informations # a list of pcds
 
    def get_target_pc(self, mesh_path, num_sample):
        mesh = trimesh.load(mesh_path)
        return mesh.sample(num_sample)

    def get_input_for_icp(
            self, pcds, voxel_size, 
            target_mesh_path, get_base,
            sample_point_number=1000):
        print(":: Load two point clouds and disturb initial pose.")
        source_raw = o3d.geometry.PointCloud()
        source_raw.points = o3d.utility.Vector3dVector(np.vstack(pcds))
        source_inlier = remove_outlier(source_raw, nb_neighbors=20, std_ratio=0.3)
        if get_base : source_all = get_pcd_with_base(source_inlier)
        else : source_all = source_inlier
        source_pc = farthest_point_sampling(source_all.points, sample_point_number)
        source_pc.paint_uniform_color([0.5, 0., 0])
        target_all = self.get_target_pc(target_mesh_path, sample_point_number*10)
        target_pc = farthest_point_sampling(target_all, sample_point_number)
        print(target_pc)
        trans_init = np.eye(4)
        source_pc.transform(trans_init)
        print(source_pc)
        self.draw_registration_result(source_pc, target_pc, np.identity(4))
        source_down, source_fpfh = preprocess_point_cloud(source_pc, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(target_pc, voxel_size)

        return source_pc, target_pc, source_down, target_down, source_fpfh, target_fpfh

    def get_T_world_wrt_object(self, pcds, est_const, target_mesh_path,
                               sample_point_number=500,get_base = True, max_iter=10):
        voxel_size = 0.2
        source, target, source_down, target_down, source_fpfh, target_fpfh = self.get_input_for_icp(
                                                            pcds, voxel_size, target_mesh_path, get_base,
                                                            sample_point_number=sample_point_number)
        for i in range(max_iter):
            result_ransac = execute_global_registration(source_down, target_down,
                                                        source_fpfh, target_fpfh,
                                                        voxel_size)
            result_icp = refine_registration(source, target, result_ransac, voxel_size)
            print(f"fitness:{result_icp.fitness}, rmse:{result_icp.inlier_rmse}")
            if result_icp.fitness > est_const[0] and result_icp.inlier_rmse < est_const[1]:
                print(result_icp)
                break
        self.draw_registration_result(source, target, result_icp.transformation)
        return result_icp.transformation
    
    ## visualization
    def visualize_scene(self, pcds, frame_size=0.1):
        colors = [[0,0,1], [1,0,0],[0,1,0]]
        pcd_o3ds = []
        frames = []
        for i, cam in enumerate(self.cams):
            pcd_o3d_raw= o3d.geometry.PointCloud()
            pcd_o3d_raw.points = o3d.utility.Vector3dVector(pcds[i])
            pcd_o3d_no_base = remove_outlier(pcd_o3d_raw, nb_neighbors=20, std_ratio=0.3)
            pcd_o3d = get_pcd_with_base(pcd_o3d_no_base)
            pcd_o3d.paint_uniform_color(colors[i])             
            pcd_o3ds.append(pcd_o3d)
            
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
            coord.transform(cam.T_calib)
            frames.append(coord)
        coord_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        frames.append(coord_world)
        return pcd_o3ds, frames
    
    def draw_registration_result(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
        o3d.visualization.draw_geometries([source_temp, target_temp, coord])