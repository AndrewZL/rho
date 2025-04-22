#!/usr/bin/env python3

import heapq
import math
import random

import numpy as np

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid

from rob498.utils import *
from typing import Dict, List, Tuple, Optional


class Planner:
    """
    A class to plan the next best view for a drone using occupancy grid mapping and A* pathfinding.
    The class handles the processing of AprilTag detections, occupancy grid updates, and pathfinding to targets.
    """

    def __init__(self, logger):
        self.logger = logger

        # Setup occupancy grid: 6m x 6m, 10cm resolution.
        self.occ_grid = OccupancyGrid()
        self.occ_grid.header.frame_id = "map"
        self.occ_grid.info.resolution = 0.1
        self.occ_grid.info.width = 60
        self.occ_grid.info.height = 60
        self.occ_grid.info.origin.position.x = -3.0
        self.occ_grid.info.origin.position.y = -6.0
        self.occ_grid.info.origin.position.z = 0.0
        self.occ_grid.info.origin.orientation.w = 1.0
        self.occ_grid.info.origin.orientation.x = 0.0
        self.occ_grid.info.origin.orientation.y = 0.0
        self.occ_grid.info.origin.orientation.z = 0.0
        # Initialize all cells to unknown (0)
        self.occ_grid.data = [0] * (self.occ_grid.info.width * self.occ_grid.info.height)

        # Shorthand
        self.origin = self.occ_grid.info.origin.position
        self.resolution = self.occ_grid.info.resolution

        # Store obstacles as grid coordinates.
        self.visited_tags = set()
        self.obstacle_locations = []
        self.targets = []
        self.visited_targets = set()

        # Camera parameters
        self.fov = 90  # Field of view in degrees
        self.max_range = 3.0  # Maximum range (meters) for free-space update

    def to_grid_coordinates(self, pos_x: float, pos_y: float) -> Tuple[int, int]:
        """
        Convert a position in meters to grid coordinates.
        """
        x = int((pos_x - self.origin.x) / self.resolution)
        y = int((pos_y - self.origin.y) / self.resolution)
        return x, y

    def process_detections(self, drone_pose: PoseStamped, detections: List[dict]) -> bool:
        """
        Process the AprilTag detections
        """
        if not detections:
            self.logger.info("No detections in message.")
            return True

        for detection in detections:
            tag_id = detection["detection"].tag_id
            if tag_id in self.visited_tags:
                self.logger.info(f"Tag ID {tag_id} already visited.")
                continue

            detection_pos = detection["tvec"]

            # Tag position in camera frame
            T_camera_tag = np.eye(4)
            T_camera_tag[:3, 3] = [detection_pos[0], detection_pos[1], detection_pos[2]]
            T_camera_tag[3, 3] = 1.0
            self.logger.info(f"T_camera_tag {T_camera_tag}")

            # Transform to drone frame
            T_drone_camera = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

            # Transform to map frame
            R_map_drone = quaternion_to_rotation_matrix(
                [drone_pose.pose.orientation.x, drone_pose.pose.orientation.y, drone_pose.pose.orientation.z, drone_pose.pose.orientation.w]
            )
            T_map_drone = np.eye(4)
            T_map_drone[:3, :3] = R_map_drone
            T_map_drone[:3, 3] = [drone_pose.pose.position.x, drone_pose.pose.position.y, drone_pose.pose.position.z]
            T_map_drone[3, 3] = 1.0
            self.logger.info(f"T_map_drone {T_map_drone}")

            T_map_tag = T_map_drone @ T_drone_camera @ T_camera_tag
            detection_pos = T_map_tag[:3, 3]

            detection_pose = PoseStamped()
            detection_pose.pose.position.x = detection_pos[0]
            detection_pose.pose.position.y = detection_pos[1]
            detection_pose.pose.position.z = detection_pos[2]

            # tag_id 0-4 are obstacle, 5-11 are targets
            if tag_id >= 5:
                self.visited_tags.add(tag_id)
                self.targets.append((detection_pose, tag_id))
                self.logger.info("Target {} added".format(tag_id))
            else:
                self.visited_tags.add(tag_id)
                self.obstacle_locations.append(detection_pose)
                self.logger.info("Obstacle {} added".format(tag_id))

        if len(self.obstacle_locations) < 3:
            self.logger.info(f"Did not find all obstacles: {self.visited_tags}")
            return False
        return True

    def update_occupancy_grid(self, drone_pose: PoseStamped):
        """
        Update the occupancy grid based on obstacles and free-space ray casting.
        Obstacles are expanded by a safety radius.
        """
        # Mark obstacles (with a safety boundary of 8 cells = 80 cm).
        safety_radius = 8
        for obstacle in self.obstacle_locations:
            obs_x, obs_y = self.to_grid_coordinates(obstacle.pose.position.x, obstacle.pose.position.y)
            for i in range(-safety_radius, safety_radius + 1):
                for j in range(-safety_radius, safety_radius + 1):
                    cell_x = obs_x + i
                    cell_y = obs_y + j
                    if 0 <= cell_x < self.occ_grid.info.width and 0 <= cell_y < self.occ_grid.info.height:
                        idx = cell_y * self.occ_grid.info.width + cell_x
                        self.occ_grid.data[idx] = 127  # 100% certainty of occupancy

        # Update free space using ray casting.
        drone_pos_x, drone_pos_y = drone_pose.pose.position.x, drone_pose.pose.position.y
        drone_grid_x, drone_grid_y = self.to_grid_coordinates(drone_pos_x, drone_pos_y)
        drone_yaw = get_yaw(drone_pose.pose.orientation)
        half_fov = self.fov / 2.0
        num_rays = 50

        for i in range(num_rays + 1):
            angle_deg = -half_fov + i * (self.fov / num_rays)
            angle_rad = math.radians(angle_deg) + drone_yaw
            ray_end_x = drone_pos_x + self.max_range * math.cos(angle_rad)
            ray_end_y = drone_pos_y + self.max_range * math.sin(angle_rad)
            end_grid_x, end_grid_y = self.to_grid_coordinates(ray_end_x, ray_end_y)
            ray_cells = bresenham_line(drone_grid_x, drone_grid_y, end_grid_x, end_grid_y)
            for cell_x, cell_y in ray_cells:
                if 0 <= cell_x < self.occ_grid.info.width and 0 <= cell_y < self.occ_grid.info.height:
                    idx = cell_y * self.occ_grid.info.width + cell_x
                    if self.occ_grid.data[idx] == 127:
                        break  # Stop the ray if an obstacle is encountered.
                    # Otherwise, mark free space with a negative value (scaled free probability).
                    cell_world_x = self.occ_grid.info.origin.position.x + (cell_x + 0.5) * self.occ_grid.info.resolution
                    cell_world_y = self.occ_grid.info.origin.position.y + (cell_y + 0.5) * self.occ_grid.info.resolution
                    d = math.hypot(cell_world_x - drone_pos_x, cell_world_y - drone_pos_y)
                    free_prob = 1 - (d / 5) ** 1.5
                    self.occ_grid.data[idx] = min(self.occ_grid.data[idx], int(-128 * free_prob))

    def cell_probability(self, cell_value: int) -> float:
        """
        Convert a cell value into an occupancy probability continuously.
        p = (cell_value + 128) / 255.
        """
        return (cell_value + 128) / 255.0

    def simulate_view_entropy(self, pose: PoseStamped) -> float:
        """
        Simulate the view entropy (information gain) for a given camera pose by
        casting rays from the pose and summing Shannon entropy over free cells.
        Returns the total entropy.
        """
        drone_x = pose.pose.position.x
        drone_y = pose.pose.position.y
        drone_grid_x, drone_grid_y = self.to_grid_coordinates(drone_x, drone_y)
        # Check if the drone is within the grid.
        if drone_grid_x < 0 or drone_grid_x >= self.occ_grid.info.width or drone_grid_y < 0 or drone_grid_y >= self.occ_grid.info.height:
            return 0.0
        # If the drone is in an occupied cell, no entropy is gained.
        idx = drone_grid_y * self.occ_grid.info.width + drone_grid_x
        if self.occ_grid.data[idx] == 127:
            return 0.0

        yaw = get_yaw(pose.pose.orientation)
        half_fov = self.fov / 2.0
        num_rays = 20
        total_entropy = 0.0

        for i in range(num_rays + 1):
            angle_deg = -half_fov + i * (self.fov / num_rays)
            angle_rad = math.radians(angle_deg) + yaw
            ray_end_x = drone_x + self.max_range * math.cos(angle_rad)
            ray_end_y = drone_y + self.max_range * math.sin(angle_rad)
            end_grid_x = int((ray_end_x - self.occ_grid.info.origin.position.x) / self.occ_grid.info.resolution)
            end_grid_y = int((ray_end_y - self.occ_grid.info.origin.position.y) / self.occ_grid.info.resolution)
            ray_cells = bresenham_line(drone_grid_x, drone_grid_y, end_grid_x, end_grid_y)
            for cell_x, cell_y in ray_cells:
                if 0 <= cell_x < self.occ_grid.info.width and 0 <= cell_y < self.occ_grid.info.height:
                    idx = cell_y * self.occ_grid.info.width + cell_x
                    if self.occ_grid.data[idx] == 127:
                        break
                    p = self.cell_probability(self.occ_grid.data[idx])
                    total_entropy += shannon_entropy(p)
        return total_entropy

    def next_best_view(
        self, current_pose: PoseStamped, num_samples: int = 50, sample_radius: float = 2.0, safe_threshold: int = -10, local_pose_pub=None, pose_des=None
    ) -> Optional[PoseStamped]:
        """
        Compute a next-best-view pose by randomly sampling candidate poses and
        selecting the one with the highest simulated view entropy.
        Only candidates within safe occupancy zones are allowed, defined by the safe_threshold.
        """
        self.logger.info("computing next best view")
        best_entropy = -1.0
        best_candidate = None

        for _ in range(num_samples):
            # hack: publish to keep the drone in the air
            local_pose_pub.publish(pose_des)

            angle_offset = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, sample_radius)

            candidate_pose = PoseStamped()
            candidate_pose.header.frame_id = "map"
            candidate_pose.pose.position.x = current_pose.pose.position.x + distance * math.cos(angle_offset)
            candidate_pose.pose.position.y = current_pose.pose.position.y + distance * math.sin(angle_offset)
            candidate_pose.pose.position.z = current_pose.pose.position.z

            candidate_yaw = random.uniform(-math.pi, math.pi)
            candidate_pose.pose.orientation.x = 0.0
            candidate_pose.pose.orientation.y = 0.0
            candidate_pose.pose.orientation.z = math.sin(candidate_yaw / 2)
            candidate_pose.pose.orientation.w = math.cos(candidate_yaw / 2)

            # Convert to grid coordinates
            grid_x, grid_y = self.to_grid_coordinates(candidate_pose.pose.position.x, candidate_pose.pose.position.y)

            if 0 <= grid_x < self.occ_grid.info.width and 0 <= grid_y < self.occ_grid.info.height:
                idx = grid_y * self.occ_grid.info.width + grid_x
                cell_value = self.occ_grid.data[idx]

                # Check safety based on occupancy threshold
                if cell_value > safe_threshold:
                    continue

                # Optional: reject candidates too close to unknown zones (3x3 patch)
                safe = True
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        nx, ny = grid_x + dx, grid_y + dy
                        if 0 <= nx < self.occ_grid.info.width and 0 <= ny < self.occ_grid.info.height:
                            nidx = ny * self.occ_grid.info.width + nx
                            if self.occ_grid.data[nidx] > safe_threshold:
                                safe = False
                                break
                    if not safe:
                        break
                if not safe:
                    continue

                # Simulate entropy
                view_entropy = self.simulate_view_entropy(candidate_pose)
                if view_entropy > best_entropy:
                    best_entropy = view_entropy
                    best_candidate = candidate_pose

        return best_candidate

    def go_to_home(self, current_pose: PoseStamped) -> Optional[List[Tuple[float, float]]]:
        """
        Go to a predefined home position (30, 55) in grid coordinates.
        """
        start_cell = (int((current_pose.pose.position.x - self.origin.x) / self.resolution), int((current_pose.pose.position.y - self.origin.y) / self.resolution))
        target_cell = (30, 55)

        path = a_star(start_cell, target_cell, self.occ_grid.data, self.occ_grid.info.width, self.occ_grid.info.height)
        self.logger.info(f"path {path}")
        if path is not None:
            self.logger.info("Visiting a Target!")
            path_coords = [(self.origin.x + (gx + 0.5) * self.resolution, self.origin.y + (gy + 0.5) * self.resolution) for gx, gy in path]
        return path_coords

    def get_next_poses(self, current_pose: PoseStamped, local_pose_pub=None, pose_des=None) -> Tuple[Optional[List[Tuple[float, float, float]]], bool]:
        """
        Compute the next best camera pose and the list of waypoints leading to it.
        Updates the occupancy grid based on the current pose and plans a safe path.
        """
        self.update_occupancy_grid(current_pose)
        local_pose_pub.publish(pose_des)

        next_pose = None
        path_coords = None
        is_target = False

        start_cell = (int((current_pose.pose.position.x - self.origin.x) / self.resolution), int((current_pose.pose.position.y - self.origin.y) / self.resolution))

        # Check for visible targets
        unvisited_targets = []
        for target, id in self.targets:
            if id not in self.visited_targets:
                unvisited_targets.append((target, id))
        self.logger.info(f"unvisited targets {unvisited_targets}")

        if unvisited_targets:
            unvisited_targets.sort(key=lambda t: math.hypot(t[0].pose.position.x - current_pose.pose.position.x, t[0].pose.position.y - current_pose.pose.position.y))
            chosen_target, chosen_id = unvisited_targets[0]
            self.logger.info(f"trying to visit {chosen_id}")
            target_cell = self.to_grid_coordinates(chosen_target.pose.position.x, chosen_target.pose.position.y)

            # hack: publish to keep the drone in the air
            local_pose_pub.publish(pose_des)

            path = a_star(start_cell, target_cell, self.occ_grid.data, self.occ_grid.info.width, self.occ_grid.info.height)

            local_pose_pub.publish(pose_des)

            self.logger.info(f"path {path}")
            if path is not None:
                self.logger.info("Visiting a Target!")
                path_coords = [(self.origin.x + (gx + 0.5) * self.resolution, self.origin.y + (gy + 0.5) * self.resolution) for gx, gy in path]
                optimized_pose, _ = self.optimize_camera_orientation(chosen_target, local_pose_pub, pose_des)
                # interpolate theta between current and optimized pose and add to path_coords
                theta_current = get_yaw(current_pose.pose.orientation)
                theta_optimized = get_yaw(optimized_pose.pose.orientation)
                theta_interpolated = np.linspace(theta_current, theta_optimized, num=len(path_coords))
                for i in range(len(path_coords)):
                    x, y = path_coords[i]
                    theta = theta_interpolated[i]
                    path_coords[i] = (x, y, theta)
                next_pose = optimized_pose
                is_target = True
                self.visited_targets.add(chosen_id)

        if next_pose is None:
            self.logger.info("trying to visit nbv")
            for safe_threshold, radius, attempts in [(-80, 4.0, 10), (-30, 3.0, 5), (-20, 1.0, 3)]:
                for _ in range(attempts):
                    nbv_candidate = self.next_best_view(
                        current_pose, num_samples=2000, sample_radius=radius, safe_threshold=safe_threshold, local_pose_pub=local_pose_pub, pose_des=pose_des
                    )
                    if nbv_candidate is None:
                        continue
                    candidate_cell = (
                        int((nbv_candidate.pose.position.x - self.origin.x) / self.resolution),
                        int((nbv_candidate.pose.position.y - self.origin.y) / self.resolution),
                    )
                    self.logger.info("trying to compute a path")
                    local_pose_pub.publish(pose_des)
                    path = a_star(start_cell, candidate_cell, self.occ_grid.data, self.occ_grid.info.width, self.occ_grid.info.height, safe_threshold=safe_threshold)
                    local_pose_pub.publish(pose_des)

                    if path is not None:
                        self.logger.info("Visiting a NBV")
                        next_pose = nbv_candidate
                        path_coords = [(self.origin.x + (gx + 0.5) * self.resolution, self.origin.y + (gy + 0.5) * self.resolution) for gx, gy in path]
                        theta_current = get_yaw(current_pose.pose.orientation)
                        theta_optimized = get_yaw(next_pose.pose.orientation)
                        theta_interpolated = np.linspace(theta_current, theta_optimized, num=len(path_coords))
                        for i in range(len(path_coords)):
                            x, y = path_coords[i]
                            theta = theta_interpolated[i]
                            path_coords[i] = (x, y, theta)
                        break
                if next_pose is not None:
                    break

        if next_pose is None:
            path_coords = None
            is_target = False

        local_pose_pub.publish(pose_des)

        # Logging: append occupancy grid data to file
        with open("occupancy_grid.txt", "a") as f:
            f.write(f"\nOccupancy grid at {current_pose.pose.position.x}, {current_pose.pose.position.y}:\n")
            for i in range(self.occ_grid.info.height):
                for j in range(self.occ_grid.info.width):
                    f.write(f"{self.occ_grid.data[i * self.occ_grid.info.width + j]} ")
                f.write("\n")
        local_pose_pub.publish(pose_des)

        with open("visited_targets.txt", "a") as f:
            f.write(f"Visited targets: {self.visited_targets}\n")
        local_pose_pub.publish(pose_des)

        with open("obstacle_locations.txt", "a") as f:
            f.write(f"Obstacle locations: {self.obstacle_locations}\n")
        local_pose_pub.publish(pose_des)

        with open("targets.txt", "a") as f:
            f.write(f"Targets: {self.targets}\n")
        local_pose_pub.publish(pose_des)

        with open("next_pose.txt", "a") as f:
            f.write(f"Next pose: {next_pose}\n")
        local_pose_pub.publish(pose_des)

        with open("path_coords.txt", "a") as f:
            f.write(f"Path: {path_coords}\n")
        local_pose_pub.publish(pose_des)

        return path_coords, is_target

    def optimize_camera_orientation(self, pose: PoseStamped, local_pose_pub=None, pose_des=None) -> Tuple[PoseStamped, float]:
        """
        At a given target pose (position fixed), sample multiple camera orientations
        and return the pose with the maximum view entropy.
        """
        best_entropy = -1.0
        best_pose = None
        # Sample 36 candidate orientations (every 10 degrees).
        for angle in np.linspace(-math.pi, math.pi, num=36, endpoint=False):
            local_pose_pub.publish(pose_des)
            candidate_pose = PoseStamped()
            candidate_pose.header.frame_id = "map"
            candidate_pose.pose.position.x = pose.pose.position.x
            candidate_pose.pose.position.y = pose.pose.position.y
            candidate_pose.pose.position.z = pose.pose.position.z
            candidate_pose.pose.orientation.x = 0.0
            candidate_pose.pose.orientation.y = 0.0
            candidate_pose.pose.orientation.z = math.sin(angle / 2)
            candidate_pose.pose.orientation.w = math.cos(angle / 2)
            entropy = self.simulate_view_entropy(candidate_pose)
            if entropy > best_entropy:
                best_entropy = entropy
                best_pose = candidate_pose
        return best_pose, best_entropy


def prune_path(
    path: List[Tuple[int, int]],
    grid: Optional[List[int]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    safe_threshold: Optional[int] = None,
    bresenham_fn: Optional[callable] = None,
) -> List[Tuple[int, int]]:
    """
    Prune a path by:
    1. Removing collinear points.
    2. Shortcutting over intermediate points when obstacle-free.
    """
    if len(path) <= 2:
        return path

    # Step 1: Remove collinear points
    collinear_pruned = [path[0]]
    for i in range(1, len(path) - 1):
        prev = collinear_pruned[-1]
        curr = path[i]
        nxt = path[i + 1]

        dx1, dy1 = curr[0] - prev[0], curr[1] - prev[1]
        dx2, dy2 = nxt[0] - curr[0], nxt[1] - curr[1]

        if dx1 * dy2 != dx2 * dy1:  # Not collinear
            collinear_pruned.append(curr)
    collinear_pruned.append(path[-1])

    # Step 2: Shortcut using line-of-sight
    if grid is None or width is None or height is None or safe_threshold is None or bresenham_fn is None:
        return collinear_pruned  # Skip shortcutting if data is missing

    shortcut_path = [collinear_pruned[0]]
    i = 0
    while i < len(collinear_pruned) - 1:
        j = len(collinear_pruned) - 1
        while j > i + 1:
            line = bresenham_fn(collinear_pruned[i][0], collinear_pruned[i][1], collinear_pruned[j][0], collinear_pruned[j][1])
            if all(0 <= x < width and 0 <= y < height and grid[y * width + x] <= safe_threshold for x, y in line):
                break
            j -= 1
        shortcut_path.append(collinear_pruned[j])
        i = j

    return shortcut_path


def a_star(start: Tuple[int, int], goal: Tuple[int, int], grid: List[int], width: int, height: int, safe_threshold: int = -1) -> Optional[List[Tuple[int, int]]]:
    """
    A* pathfinding on a 2D occupancy grid with 8-connected movement.
    Avoids cells above safe_threshold. Includes pruning to remove redundant waypoints.
    """

    def heuristic(a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])

    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    open_set: List[Tuple[float, Tuple[int, int]]] = []
    heapq.heappush(open_set, (0, start))
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score = {start: 0.0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            # Reconstruct full path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return prune_path(path, grid=grid, width=width, height=height, safe_threshold=safe_threshold, bresenham_fn=bresenham_line)

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)

            if 0 <= neighbor[0] < width and 0 <= neighbor[1] < height:
                idx = neighbor[1] * width + neighbor[0]
                if grid[idx] > safe_threshold:
                    continue  # Skip unsafe or unknown cells

                tentative_g = g_score[current] + math.hypot(dx, dy)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

    return None  # No valid path found
