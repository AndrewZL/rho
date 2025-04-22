import argparse

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize RGBD data as a point cloud.")
    parser.add_argument("file_path", type=str, help="Path to the .npy file containing RGBD data.")
    args = parser.parse_args()

    fp = args.file_path

    rgbd_data = np.load(fp)  # Load your .npy file containing RGBD data

    color_image = rgbd_data[..., :3]  # RGB values
    depth_image = rgbd_data[..., 3]  # Depth values (ensure it's in meters)

    depth_image[depth_image > 1] = 1
    depth_image[depth_image < 0] = 0

    fx = 286.1167907714844  # focal length in x 
    fy = 286.1167907714844  # focal length in y 
    cx = depth_image.shape[0] / 2  # principal point x 752 / 2
    cy = depth_image.shape[1] / 2  # principal point y 200 / 2

    height, width = depth_image.shape
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    X = (xx - cx) * depth_image / fx
    Y = (yy - cy) * depth_image / fy
    Z = depth_image

    points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(color_image.reshape(-1, 3) / 255.0)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])
