import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
import cv2
import open3d as o3d

# File paths of custom RGB and depth images in JPG format
custom_color_path = '1.jpg'
custom_depth_path = '1.png'

# Load custom RGB and depth images
color_raw = o3d.io.read_image(custom_color_path)
depth_raw = o3d.io.read_image(custom_depth_path)

# Create Open3D Image objects from the loaded images
color = o3d.geometry.Image(color_raw)
depth = o3d.geometry.Image(depth_raw)

# Create RGBDImage from custom color and depth images
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

o3d.visualization.draw_geometries([pcd])




# Estimate normals for the point cloud
pcd.estimate_normals()

# Set the plane distance threshold for plane segmentation (adjust as needed)
plane_distance_threshold = 0.01

# Perform plane segmentation
_, inliers = pcd.segment_plane(distance_threshold=plane_distance_threshold,
                                ransac_n=3,
                                num_iterations=1000)

# Extract the inlier points (floor points)
floor_points = pcd.select_by_index(inliers)

# Set the color of the floor points to red
red_color = [1, 0, 0]  # Red color
floor_points.paint_uniform_color(red_color)

# Visualize the point cloud with the colored floor points
o3d.visualization.draw_geometries([pcd])