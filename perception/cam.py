import numpy as np
import cv2
import sys
import zivid
from zivid.experimental import calibration
from pathlib import Path
import pyrealsense2 as rs


class Camera:
    def __init__(self, idx):
        self.idx = idx
        self.T_calib = None
    def get_rgb_pcd(self):
        pass
    def get_intrinsics(self):
        pass
    def normalize_image(self, img, aspect_ratio=4/3, dsize=(640,480)):
        if len(img.shape) == 3:
            height, width, _ = img.shape
        else:
            height, width = img.shape
        desired_x_len = height*aspect_ratio  
        crop_len = int((width - desired_x_len)/2)
        img = cv2.resize(img[:,crop_len:width-crop_len], 
                         dsize=dsize, 
                         interpolation=cv2.INTER_AREA)
        return img
    
    def calibrate_from_file(self, config_path="./config"):
        path = Path(config_path)
        self.T_calib = np.load(path/f"cam{self.idx}.npy")

    def calibrate(self, force=False, save=True, config_path="./config", wait=3):
        path = Path(config_path) / f"cam{self.idx}.npy"
        if path.exists() and (not force):
            self.calibrate_from_file()
            return

        CHECKERBOARD = (4,5)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        x = np.repeat([-2, -1, 0, 1, 2], 4)
        y = np.tile([-1, 0, 1, 2], 5)
        z = np.zeros(20)
        world_points = np.vstack([x, y, z]).T * 0.03
        imgpoints = []
        img, _ = self.get_rgb_pcd()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray,
                                                 CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            imgpoints.append(corners2)
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            for i in range(0,len(corners2)):
                img=cv2.putText(
                    img, str(i), (int(corners2[i,0,0]),int(corners2[i,0,1])),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),1,2)
            cv2.imshow('img',img)
            cv2.waitKey(wait*1000)
        cv2.destroyAllWindows()
                
        cam_mtx, cam_dist = self.get_intrinsics()
        ret, rvec, tvec = cv2.solvePnP(world_points, imgpoints[0], cam_mtx, cam_dist)
        rmat, _ = cv2.Rodrigues(rvec)
        
        # calibration result
        self.T_calib = np.linalg.inv(np.r_[np.concatenate((rmat, tvec),1), [[0,0,0,1]]])
        
        if save:
            np.save(path, self.T_calib)

class Zivid(Camera):
    IMG_SIZE = (1944, 1200)
    def __init__(self, idx, client, settings):
        self.client = client
        self.settings = settings
        super().__init__(idx=idx)

    def get_rgb_pcd(self):
        self.client.connect()
        frame = self.client.capture(self.settings)
        rgba = frame.point_cloud().copy_data("rgba")
        pcd = frame.point_cloud().copy_data("xyz")
        nan_idx = np.isnan(pcd[:,:,0])
        pcd[nan_idx,:] = 0
        pcd = pcd/1000
        self.client.disconnect()
        rgb = self.normalize_image(rgba[:,:,:3])
        pcd = self.normalize_image(pcd)
        return rgb, pcd
    
    def get_intrinsics(self):
        RESIZE_SCALE = 0.4
        CENTER_TO_EDGE = 172
        self.client.connect()
        cam_intrinsics = calibration.intrinsics(self.client, self.settings)
        mtx = cam_intrinsics.camera_matrix
        dist = cam_intrinsics.distortion
        cam_matrix = np.asmatrix([[mtx.fx*RESIZE_SCALE, 0, mtx.cx*RESIZE_SCALE-CENTER_TO_EDGE*RESIZE_SCALE],
                                  [0, mtx.fy*RESIZE_SCALE, mtx.cy*RESIZE_SCALE],
                                  [0, 0, 1]])
        cam_distortion = (dist.k1, dist.k2, dist.k3, dist.p1, dist.p2)
        self.client.disconnect()
        return cam_matrix, cam_distortion

class L515(Camera):
    def __init__(self, idx, serial):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)
        self.pipeline.start(self.config)
        super().__init__(idx=idx)

    def get_aligned_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        return aligned_frames
    
    def get_rgb_pcd(self):
        cam_mtx, _ = self.get_intrinsics()
        frame_depth = np.asanyarray(self.get_aligned_frames().get_depth_frame().get_data())*0.25  # depth scale: 4X
        frame_color = np.asanyarray(self.get_aligned_frames().get_color_frame().get_data())
        rgb = self.normalize_image(frame_color)
        depth = self.normalize_image(frame_depth)
        z= depth.astype(float)/1000
        height, width = depth.shape
        px, py = np.meshgrid(np.arange(width), np.arange(height))
        px, py = px.astype(float), py.astype(float)
        x = ((px - cam_mtx[0, 2]) / cam_mtx[0, 0]) * z
        y = ((py - cam_mtx[1, 2]) / cam_mtx[1, 1]) * z
        pcd = np.concatenate([i[..., np.newaxis] for i in (x, y, z)], axis=-1)
        return rgb, pcd
    
    def get_intrinsics(self):
        """
        Using modified intrinsics because of image cropping and resizing
        """
        RESIZE_SCALE = 8/9
        CENTER_TO_EDGE = 120
        cam_intr = self.get_aligned_frames().get_color_frame().profile.as_video_stream_profile().intrinsics
        cam_matrix = np.asmatrix([
            [cam_intr.fx*RESIZE_SCALE, 0, cam_intr.ppx*RESIZE_SCALE-CENTER_TO_EDGE*RESIZE_SCALE],
            [0, cam_intr.fy*RESIZE_SCALE, cam_intr.ppy*RESIZE_SCALE],
            [0, 0, 1]])
        cam_distortion = cam_intr.coeffs
        return cam_matrix, tuple(cam_distortion)
