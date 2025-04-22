#!/usr/bin/env python3
"""
This script simulates an occupancy-grid based exploration strategy.
It drops 10 random "targets" into the map (ensuring they are not inside obstacles),
and uses a two-phase strategy:
  1. If any unvisited target is visible (i.e. within the field-of-view and with an unobstructed line-of-sight)
     from the current camera pose, the planner computes a collision-free path (using A* or PRM) to the nearest one.
  2. Once the target is reached, the camera orientation is optimized for maximum information gain.
     The target is then marked as visited.
  3. If no targets are visible, the system reverts to the next-best-view strategy.
  
The occupancy grid, obstacles, targets (red for unvisited, blue for visited), planned path,
and camera indicator (with corrected orientation) are all plotted.
Press the right arrow key to update the planner, or Esc to exit.
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import heapq  # For A* search

### FOR USE WITH ACTUAL CAMERA POSE ####
### RETURNS THE NEXT BEST POSE AND A LIST OF WAYPOINTS ###
def get_next_poses(current_pose):
    """
    Compute the next best camera pose and the list of waypoints leading to it.
    Updates the occupancy grid based on the current pose and plans a safe path.

    Args:
        current_pose (PoseStamped): The current pose of the robot/camera.

    Returns:
        Tuple[PoseStamped, List[Tuple[float, float]]]:
            - next_pose: the selected next pose (camera view optimized)
            - path_coords: list of (x, y) waypoints including the final pose
    """
    global planner, use_prm_planner, targets

    planner.update_occupancy_grid(current_pose)

    origin = planner.occ_grid.info.origin.position
    res = planner.occ_grid.info.resolution
    grid = planner.occ_grid.data
    width = planner.occ_grid.info.width
    height = planner.occ_grid.info.height

    next_pose = None
    path_coords = None

    start_cell = (int((current_pose.pose.position.x - origin.x) / res),
                  int((current_pose.pose.position.y - origin.y) / res))

    # Check for visible targets
    visible_targets = []
    for target in targets:
        if not getattr(target, "visited", False) and is_target_visible(current_pose, target):
            visible_targets.append(target)

    if visible_targets:
        visible_targets.sort(key=lambda t: math.hypot(t.pose.position.x - current_pose.pose.position.x,
                                                      t.pose.position.y - current_pose.pose.position.y))
        chosen_target = visible_targets[0]
        target_cell = (int((chosen_target.pose.position.x - origin.x) / res),
                       int((chosen_target.pose.position.y - origin.y) / res))

        if use_prm_planner:
            path = prm_planner(start_cell, target_cell, grid, width, height, num_samples=50)
        else:
            path = a_star(start_cell, target_cell, grid, width, height)

        if path is not None:
            path_coords = [(origin.x + (gx + 0.5) * res, origin.y + (gy + 0.5) * res) for gx, gy in path]
            optimized_pose, _ = optimize_camera_orientation(chosen_target)
            next_pose = optimized_pose
            chosen_target.visited = True

    if next_pose is None:
        for safe_threshold, radius, attempts in [(-80, 4.0, 10), (-30, 3.0, 5), (-20, 1.0, 3)]:
            for _ in range(attempts):
                nbv_candidate = planner.next_best_view(current_pose, num_samples=2000, sample_radius=radius, safe_threshold=safe_threshold)
                if nbv_candidate is None:
                    continue
                candidate_cell = (int((nbv_candidate.pose.position.x - origin.x) / res),
                                  int((nbv_candidate.pose.position.y - origin.y) / res))
                if use_prm_planner:
                    path = prm_planner(start_cell, candidate_cell, grid, width, height, num_samples=50, safe_threshold=safe_threshold)
                else:
                    path = a_star(start_cell, candidate_cell, grid, width, height, safe_threshold=safe_threshold)
                if path is not None:
                    next_pose = nbv_candidate
                    path_coords = [(origin.x + (gx + 0.5) * res, origin.y + (gy + 0.5) * res) for gx, gy in path]
                    break
            if next_pose is not None:
                break

    if next_pose is None:
        next_pose = current_pose
        path_coords = [(current_pose.pose.position.x, current_pose.pose.position.y)]

    return next_pose, path_coords

# For testing without ROS installed, we provide simple mock classes.
# Remove these mocks when running in a proper ROS environment.
class Pose:
    def __init__(self):
        self.position = type("obj", (), {"x": 0.0, "y": 0.0, "z": 0.0})
        self.orientation = type("obj", (), {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0})

class PoseStamped:
    def __init__(self):
        self.header = type("obj", (), {"frame_id": "map", "stamp": None})
        self.pose = Pose()

class OccupancyGrid:
    def __init__(self):
        self.header = type("obj", (), {"frame_id": "map", "stamp": None})
        self.info = type("obj", (), {
            "resolution": 0.1,
            "width": 40,
            "height": 40,
            "origin": Pose()
        })
        self.data = []

# --- Helper function: convert quaternion to Euler angles ---
def quaternion_to_euler(q):
    """
    Convert a quaternion [x, y, z, w] to Euler angles (roll, pitch, yaw).
    """
    x, y, z, w = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

def shannon_entropy(p):
    """Compute Shannon's entropy (in bits) for a probability p."""
    if p == 0 or p == 1:
        return 0
    return - (p * math.log2(p) + (1 - p) * math.log2(1 - p))

class Planner():
    def __init__(self): 
        # Setup occupancy grid: 4m x 4m, 10cm resolution.
        self.occ_grid = OccupancyGrid()
        self.occ_grid.header.frame_id = "map"
        self.occ_grid.info.resolution = 0.1
        self.occ_grid.info.width = 40
        self.occ_grid.info.height = 40
        self.occ_grid.info.origin.position.x = -2.0
        self.occ_grid.info.origin.position.y = -2.0
        self.occ_grid.info.origin.position.z = 0.0
        self.occ_grid.info.origin.orientation.w = 1.0
        self.occ_grid.info.origin.orientation.x = 0.0
        self.occ_grid.info.origin.orientation.y = 0.0
        self.occ_grid.info.origin.orientation.z = 0.0
        # Initialize all cells to unknown (0)
        self.occ_grid.data = [0] * (self.occ_grid.info.width * self.occ_grid.info.height)

        # Store obstacles as grid coordinates.
        self.obstacle_locations = []

        self.fov = 90        # Field of view in degrees
        self.max_range = 3.0  # Maximum range (meters) for free-space update

    def transform_to_global(self, drone_pose: PoseStamped, local_pose: PoseStamped) -> PoseStamped:
        """
        Convert a local obstacle pose into the global map frame.
        """
        global_pose = PoseStamped()
        global_pose.header.frame_id = "map"
        global_pose.header.stamp = drone_pose.header.stamp
        global_pose.pose.position.x = drone_pose.pose.position.x + local_pose.pose.position.x
        global_pose.pose.position.y = drone_pose.pose.position.y + local_pose.pose.position.y
        global_pose.pose.position.z = drone_pose.pose.position.z + local_pose.pose.position.z
        return global_pose

    def get_yaw(self, orientation) -> float:
        """
        Return yaw from a quaternion orientation.
        """
        _, _, yaw = quaternion_to_euler([orientation.x, orientation.y, orientation.z, orientation.w])
        return yaw

    def add_obstacle(self, drone_pose: PoseStamped, obstacle_local: PoseStamped):
        """
        Add an obstacle given in a local frame by transforming it to global coordinates,
        then converting to grid coordinates.
        """
        global_obstacle = self.transform_to_global(drone_pose, obstacle_local)
        grid_x = int((global_obstacle.pose.position.x - self.occ_grid.info.origin.position.x) / self.occ_grid.info.resolution)
        grid_y = int((global_obstacle.pose.position.y - self.occ_grid.info.origin.position.y) / self.occ_grid.info.resolution)
        self.obstacle_locations.append((grid_x, grid_y))

    def update_occupancy_grid(self, drone_pose: PoseStamped):
        """
        Update the occupancy grid based on obstacles and free-space ray casting.
        Obstacles are expanded by a safety radius.
        """
        # Mark obstacles (with a safety boundary of 2 cells → 20 cm).
        safety_radius = 2
        for (obs_x, obs_y) in self.obstacle_locations:
            for i in range(-safety_radius, safety_radius + 1):
                for j in range(-safety_radius, safety_radius + 1):
                    cell_x = obs_x + i
                    cell_y = obs_y + j
                    if 0 <= cell_x < self.occ_grid.info.width and 0 <= cell_y < self.occ_grid.info.height:
                        idx = cell_y * self.occ_grid.info.width + cell_x
                        self.occ_grid.data[idx] = 127  # 127 indicates certainty of occupancy

        # Update free space using ray casting.
        drone_x = drone_pose.pose.position.x
        drone_y = drone_pose.pose.position.y
        drone_grid_x = int((drone_x - self.occ_grid.info.origin.position.x) / self.occ_grid.info.resolution)
        drone_grid_y = int((drone_y - self.occ_grid.info.origin.position.y) / self.occ_grid.info.resolution)
        drone_yaw = self.get_yaw(drone_pose.pose.orientation)
        half_fov = self.fov / 2.0
        num_rays = 50

        for i in range(num_rays + 1):
            angle_deg = -half_fov + i * (self.fov / num_rays)
            angle_rad = math.radians(angle_deg) + drone_yaw
            ray_end_x = drone_x + self.max_range * math.cos(angle_rad)
            ray_end_y = drone_y + self.max_range * math.sin(angle_rad)
            end_grid_x = int((ray_end_x - self.occ_grid.info.origin.position.x) / self.occ_grid.info.resolution)
            end_grid_y = int((ray_end_y - self.occ_grid.info.origin.position.y) / self.occ_grid.info.resolution)
            ray_cells = self.bresenham_line(drone_grid_x, drone_grid_y, end_grid_x, end_grid_y)
            for (cell_x, cell_y) in ray_cells:
                if 0 <= cell_x < self.occ_grid.info.width and 0 <= cell_y < self.occ_grid.info.height:
                    idx = cell_y * self.occ_grid.info.width + cell_x
                    if self.occ_grid.data[idx] == 127:
                        break  # Stop the ray if an obstacle is encountered.
                    # Otherwise, mark free space with a negative value (scaled free probability).
                    cell_world_x = self.occ_grid.info.origin.position.x + (cell_x + 0.5) * self.occ_grid.info.resolution
                    cell_world_y = self.occ_grid.info.origin.position.y + (cell_y + 0.5) * self.occ_grid.info.resolution
                    d = math.hypot(cell_world_x - drone_x, cell_world_y - drone_y)
                    free_prob = 1 - (d/3)**1.5
                    self.occ_grid.data[idx] = min(self.occ_grid.data[idx], int(-128 * free_prob))

    def bresenham_line(self, x0, y0, x1, y1):
        """
        Compute the grid cells that form a line between (x0, y0) and (x1, y1)
        using Bresenham's algorithm.
        """
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1

        if dx > dy:
            err = dx / 2.0
            while x != x1:
                cells.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                cells.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        cells.append((x1, y1))
        return cells

    def cell_probability(self, cell_value):
        """
        Convert a cell value into an occupancy probability continuously.
        p = (cell_value + 128) / 255.
        """
        return (cell_value + 128) / 255.0

    def simulate_view_entropy(self, pose: PoseStamped) -> float:
        """
        Simulate the view entropy (information gain) for a given camera pose by
        casting rays from the pose and summing Shannon entropy over free cells.
        """
        drone_x = pose.pose.position.x
        drone_y = pose.pose.position.y
        grid_x = int((drone_x - self.occ_grid.info.origin.position.x) / self.occ_grid.info.resolution)
        grid_y = int((drone_y - self.occ_grid.info.origin.position.y) / self.occ_grid.info.resolution)
        # Check if the drone is within the grid.
        if grid_x < 0 or grid_x >= self.occ_grid.info.width or grid_y < 0 or grid_y >= self.occ_grid.info.height:
            return 0.0
        # If the drone is in an occupied cell, no entropy is gained.
        idx = grid_y * self.occ_grid.info.width + grid_x
        if self.occ_grid.data[idx] == 127:
            return 0.0

        yaw = self.get_yaw(pose.pose.orientation)
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
            ray_cells = self.bresenham_line(grid_x, grid_y, end_grid_x, end_grid_y)
            for (cell_x, cell_y) in ray_cells:
                if 0 <= cell_x < self.occ_grid.info.width and 0 <= cell_y < self.occ_grid.info.height:
                    idx = cell_y * self.occ_grid.info.width + cell_x
                    if self.occ_grid.data[idx] == 127:
                        break
                    p = self.cell_probability(self.occ_grid.data[idx])
                    total_entropy += shannon_entropy(p)
        return total_entropy

    def next_best_view(self, current_pose: PoseStamped, num_samples=50, sample_radius=2.0, safe_threshold=-10, padding=0.2) -> PoseStamped:
        """
        Compute a next-best-view pose by randomly sampling candidate poses and
        selecting the one with the highest simulated view entropy.
        Only candidates within safe occupancy zones and a padded map boundary are allowed.

        Parameters:
            current_pose (PoseStamped): The current robot/camera pose.
            num_samples (int): Number of candidate samples to try.
            sample_radius (float): Radius to sample NBV candidates within.
            safe_threshold (int): Occupancy threshold for a cell to be considered safe (≤ this value).
            padding (float): Padding (in meters) to keep away from map edges.

        Returns:
            PoseStamped: The best next pose, or None if no safe candidates were found.
        """
        def candidate_safe_margin(p, min_val, max_val):
            return max(0.0, min(p - min_val, max_val - p))

        best_entropy = -1
        best_candidate = None

        # Precompute map padded bounds
        origin = self.occ_grid.info.origin.position
        resolution = self.occ_grid.info.resolution
        map_min_x = origin.x + padding
        map_min_y = origin.y + padding
        map_max_x = origin.x + self.occ_grid.info.width * resolution - padding
        map_max_y = origin.y + self.occ_grid.info.height * resolution - padding

        for _ in range(num_samples):
            angle_offset = random.uniform(0, 2 * math.pi)

            # # Dynamically shrink radius to avoid overshooting map edge
            # max_r_x = candidate_safe_margin(current_pose.pose.position.x, map_min_x, map_max_x)
            # max_r_y = candidate_safe_margin(current_pose.pose.position.y, map_min_y, map_max_y)
            # buffered_sample_radius = min(sample_radius, max_r_x, max_r_y)

            # distance = random.uniform(0, buffered_sample_radius)
            distance = random.uniform(0, sample_radius)

            candidate_pose = PoseStamped()
            candidate_pose.header.frame_id = "map"
            candidate_pose.pose.position.x = current_pose.pose.position.x + distance * math.cos(angle_offset)
            candidate_pose.pose.position.y = current_pose.pose.position.y + distance * math.sin(angle_offset)
            candidate_pose.pose.position.z = current_pose.pose.position.z

            # Reject if outside map bounds with padding
            if not (map_min_x <= candidate_pose.pose.position.x <= map_max_x and
                    map_min_y <= candidate_pose.pose.position.y <= map_max_y):
                continue

            candidate_yaw = random.uniform(-math.pi, math.pi)
            candidate_pose.pose.orientation.x = 0.0
            candidate_pose.pose.orientation.y = 0.0
            candidate_pose.pose.orientation.z = math.sin(candidate_yaw / 2)
            candidate_pose.pose.orientation.w = math.cos(candidate_yaw / 2)

            # Convert to grid coordinates
            grid_x = int((candidate_pose.pose.position.x - origin.x) / resolution)
            grid_y = int((candidate_pose.pose.position.y - origin.y) / resolution)

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

def prune_path(path, grid=None, width=None, height=None, safe_threshold=None, bresenham_fn=None):
    """
    Prune a path by:
    1. Removing collinear points.
    2. Shortcutting over intermediate points when obstacle-free.
    
    Parameters:
        path (List[Tuple[int, int]]): Path as list of (x, y) grid cells.
        grid (List[int]): Occupancy grid (flattened) [optional, required for shortcutting].
        width (int): Grid width.
        height (int): Grid height.
        safe_threshold (int): Max cell value to consider "free".
        bresenham_fn (callable): Function to generate a line between two grid cells.

    Returns:
        List[Tuple[int, int]]: Optimized path.
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
            line = bresenham_fn(collinear_pruned[i][0], collinear_pruned[i][1],
                                collinear_pruned[j][0], collinear_pruned[j][1])
            if all(0 <= x < width and 0 <= y < height and grid[y * width + x] <= safe_threshold for x, y in line):
                break
            j -= 1
        shortcut_path.append(collinear_pruned[j])
        i = j

    return shortcut_path


# --- A* Path Planning Function ---
def a_star(start, goal, grid, width, height, safe_threshold=-10):
    """
    A* pathfinding on a 2D occupancy grid with 8-connected movement.
    Avoids cells above safe_threshold. Includes pruning to remove redundant waypoints.
    
    Args:
        start (Tuple[int, int]): Starting grid cell (x, y)
        goal (Tuple[int, int]): Goal grid cell (x, y)
        grid (List[int]): Flattened occupancy grid
        width (int): Grid width
        height (int): Grid height
        safe_threshold (int): Max occupancy value for safe cell (≤ this is safe)

    Returns:
        List[Tuple[int, int]] or None: Pruned path from start to goal if found
    """

    def heuristic(a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])

    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),          (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            # Reconstruct full path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return prune_path(
                path,
                grid=grid,
                width=width,
                height=height,
                safe_threshold=safe_threshold,
                bresenham_fn=planner.bresenham_line
)


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


# --- PRM (Probabilistic Roadmap) Planner Function ---
def prm_planner(start, goal, grid, width, height, num_samples=100, safe_threshold=-10):
    """
    Plan a path using a simple Probabilistic Roadmap (PRM) that only uses nodes
    in cells that are confidently free (i.e. occupancy <= safe_threshold).
    Samples 'num_samples' free cells and connects nodes that have a clear, safe line-of-sight.
    Returns the path as a list of (x, y) grid cells, or None if no path is found.
    """
    # Sample free nodes.
    nodes = set()
    nodes.add(start)
    nodes.add(goal)
    while len(nodes) < num_samples + 2:  # include start and goal
        sx = random.randint(0, width - 1)
        sy = random.randint(0, height - 1)
        idx = sy * width + sx
        if grid[idx] <= safe_threshold:  # cell is confidently free
            nodes.add((sx, sy))
    nodes = list(nodes)
    
    # Build a graph: connect nodes if they are close and the line-of-sight is safe.
    graph = {node: [] for node in nodes}
    for i, node in enumerate(nodes):
        for j, other in enumerate(nodes):
            if i == j:
                continue
            # Only connect if the Euclidean distance is less than a threshold.
            dist = math.hypot(other[0] - node[0], other[1] - node[1])
            if dist > 10:  # arbitrary threshold based on grid size
                continue
            # Check if line-of-sight is safe.
            line = planner.bresenham_line(node[0], node[1], other[0], other[1])
            blocked = False
            for (lx, ly) in line:
                if 0 <= lx < width and 0 <= ly < height:
                    if grid[ly * width + lx] > safe_threshold:
                        blocked = True
                        break
            if not blocked:
                graph[node].append((other, dist))
    
    # Use Dijkstra's algorithm on the graph.
    dist_so_far = {node: float('inf') for node in nodes}
    dist_so_far[start] = 0
    prev = {}
    queue = [(0, start)]
    while queue:
        current_dist, current = heapq.heappop(queue)
        if current == goal:
            break
        for neighbor, cost in graph[current]:
            alt = current_dist + cost
            if alt < dist_so_far[neighbor]:
                dist_so_far[neighbor] = alt
                prev[neighbor] = current
                heapq.heappush(queue, (alt, neighbor))
    if goal not in prev:
        return None
    # Reconstruct the path.
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path


# --- Optimize Camera Orientation at a Target ---
def optimize_camera_orientation(pose: PoseStamped) -> (PoseStamped, float):
    """
    At a given target pose (position fixed), sample multiple camera orientations
    and return the pose with the maximum view entropy.
    """
    best_entropy = -1
    best_pose = None
    # Sample 36 candidate orientations (every 10 degrees).
    for angle in np.linspace(-math.pi, math.pi, num=36, endpoint=False):
        candidate_pose = PoseStamped()
        candidate_pose.header.frame_id = "map"
        candidate_pose.pose.position.x = pose.pose.position.x
        candidate_pose.pose.position.y = pose.pose.position.y
        candidate_pose.pose.position.z = pose.pose.position.z
        candidate_pose.pose.orientation.x = 0.0
        candidate_pose.pose.orientation.y = 0.0
        candidate_pose.pose.orientation.z = math.sin(angle / 2)
        candidate_pose.pose.orientation.w = math.cos(angle / 2)
        entropy = planner.simulate_view_entropy(candidate_pose)
        if entropy > best_entropy:
            best_entropy = entropy
            best_pose = candidate_pose
    return best_pose, best_entropy

# --- Check if a Target is Visible from Current Pose ---
def is_target_visible(current_pose: PoseStamped, target: PoseStamped) -> bool:
    """
    Determine whether the target is visible from the current pose by checking:
      1. The target lies within the camera's field-of-view.
      2. There is an unobstructed line-of-sight (using Bresenham's algorithm).
    """
    dx = target.pose.position.x - current_pose.pose.position.x
    dy = target.pose.position.y - current_pose.pose.position.y
    angle_to_target = math.atan2(dy, dx)
    current_yaw = planner.get_yaw(current_pose.pose.orientation)
    # Normalize angle difference to [-pi, pi].
    diff = (angle_to_target - current_yaw + math.pi) % (2 * math.pi) - math.pi
    if abs(diff) > math.radians(planner.fov) / 2:
        return False

    # Convert positions to grid coordinates.
    grid_x1 = int((current_pose.pose.position.x - planner.occ_grid.info.origin.position.x) / planner.occ_grid.info.resolution)
    grid_y1 = int((current_pose.pose.position.y - planner.occ_grid.info.origin.position.y) / planner.occ_grid.info.resolution)
    grid_x2 = int((target.pose.position.x - planner.occ_grid.info.origin.position.x) / planner.occ_grid.info.resolution)
    grid_y2 = int((target.pose.position.y - planner.occ_grid.info.origin.position.y) / planner.occ_grid.info.resolution)
    line = planner.bresenham_line(grid_x1, grid_y1, grid_x2, grid_y2)
    for (cx, cy) in line:
        if 0 <= cx < planner.occ_grid.info.width and 0 <= cy < planner.occ_grid.info.height:
            idx = cy * planner.occ_grid.info.width + cx
            if planner.occ_grid.data[idx] == 127:
                return False  # Obstacle blocks view.
    return True

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors

# --- Helper function to add a 3D camera indicator for a given pose ---
# def add_camera_indicator_3d(ax, pose: PoseStamped, color='white', label=None, size=0.3):
#     def rotate_point(px, py, theta, cx, cy):
#         """Rotate point (px, py) around (cx, cy) by theta radians."""
#         sin_theta = math.sin(theta)
#         cos_theta = math.cos(theta)
#         nx = cos_theta * (px - cx) - sin_theta * (py - cy) + cx
#         ny = sin_theta * (px - cx) + cos_theta * (py - cy) + cy
#         return nx, ny

#     # Define a triangle in the pose's local frame
#     p1 = (pose.pose.position.x, pose.pose.position.y)  # Tip of the triangle
#     p2 = (pose.pose.position.x - size/2, pose.pose.position.y - size/4)
#     p3 = (pose.pose.position.x - size/2, pose.pose.position.y + size/4)
    
#     yaw = planner.get_yaw(pose.pose.orientation)
#     # Rotate triangle to align with the camera orientation
#     p1 = rotate_point(*p1, yaw + np.pi, pose.pose.position.x, pose.pose.position.y)
#     p2 = rotate_point(*p2, yaw + np.pi, pose.pose.position.x, pose.pose.position.y)
#     p3 = rotate_point(*p3, yaw + np.pi, pose.pose.position.x, pose.pose.position.y)
    
#     # Create 3D triangles (with small z height)
#     z_height = 0.05
    
#     # Draw the triangle edges
#     ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [z_height, z_height], color=color, linewidth=2)
#     ax.plot([p2[0], p3[0]], [p2[1], p3[1]], [z_height, z_height], color=color, linewidth=2)
#     ax.plot([p3[0], p1[0]], [p3[1], p1[1]], [z_height, z_height], color=color, linewidth=2)
    
#     # Add a line indicating the viewing direction
#     view_length = 0.4
#     view_end = (
#         pose.pose.position.x + view_length * math.cos(yaw),
#         pose.pose.position.y + view_length * math.sin(yaw)
#     )
    # ax.plot(
    #     [pose.pose.position.x, view_end[0]], 
    #     [pose.pose.position.y, view_end[1]], 
    #     [z_height, z_height], 
    #     color=color, 
    #     linewidth=2, 
    #     linestyle='--',
    #     zorder=10,
    # )


planner = None          # Instance of Planner
current_pose = None     # Current drone/camera pose (PoseStamped)
fig, ax = None, None    # Matplotlib figure and axis
targets = []            # List of target PoseStamped objects
use_prm_planner = False # Set to True to use PRM (with 50 sample points) instead of A* for planning.
num_visited = 0
# --- Function to update the 3D plot ---
def update_plot():
    global current_pose, planner, ax, fig, targets, use_prm_planner, num_visited

    previous_pose = current_pose
    planner.update_occupancy_grid(current_pose)

    origin = planner.occ_grid.info.origin.position
    res = planner.occ_grid.info.resolution
    grid = planner.occ_grid.data
    width = planner.occ_grid.info.width
    height = planner.occ_grid.info.height

    next_pose = None
    path_coords = None

    start_cell = (int((current_pose.pose.position.x - origin.x) / res),
                  int((current_pose.pose.position.y - origin.y) / res))

    # Check for visible targets
    visible_targets = []
    for target in targets:
        if not getattr(target, "visited", False) and is_target_visible(current_pose, target):
            visible_targets.append(target)

    if visible_targets:
        visible_targets.sort(key=lambda t: math.hypot(t.pose.position.x - current_pose.pose.position.x,
                                                      t.pose.position.y - current_pose.pose.position.y))
        chosen_target = visible_targets[0]
        target_cell = (int((chosen_target.pose.position.x - origin.x) / res),
                       int((chosen_target.pose.position.y - origin.y) / res))

        if use_prm_planner:
            path = prm_planner(start_cell, target_cell, grid, width, height, num_samples=50)
            planner_used = "PRM"
        else:
            path = a_star(start_cell, target_cell, grid, width, height)
            planner_used = "A*"

        if path is not None:
            path_coords = [(origin.x + (gx + 0.5) * res, origin.y + (gy + 0.5) * res) for gx, gy in path]
            optimized_pose, opt_entropy = optimize_camera_orientation(chosen_target)
            next_pose = optimized_pose
            chosen_target.visited = True
            num_visited += 1
            print(f"Visited target at ({next_pose.pose.position.x:.2f}, {next_pose.pose.position.y:.2f}) with optimized view entropy: {opt_entropy:.2f}")
        else:
            print("No valid path found to the target; skipping target.")

    if next_pose is None:
        print("No visible targets. Trying NBV.")
        # ... (continue with existing NBV logic) ...
        for safe_threshold, radius, attempts in [(-80, 4.0, 10), (-30, 3.0, 5), (-20, 1.0, 3)]:
            for _ in range(attempts):
                nbv_candidate = planner.next_best_view(current_pose, num_samples=2000, sample_radius=radius, safe_threshold=safe_threshold)
                # ... (continue with existing NBV code) ...
                if nbv_candidate is not None:
                    candidate_cell = (int((nbv_candidate.pose.position.x - origin.x) / res),
                                    int((nbv_candidate.pose.position.y - origin.y) / res))
                    if use_prm_planner:
                        path = prm_planner(start_cell, candidate_cell, grid, width, height, num_samples=50, safe_threshold=safe_threshold)
                    else:
                        path = a_star(start_cell, candidate_cell, grid, width, height, safe_threshold=safe_threshold)
                    if path is not None:
                        next_pose = nbv_candidate
                        path_coords = [(origin.x + (gx + 0.5) * res, origin.y + (gy + 0.5) * res) for gx, gy in path]
                        break
            if next_pose is not None:
                break

        if next_pose is None:
            next_pose = current_pose
            path_coords = [(current_pose.pose.position.x, current_pose.pose.position.y)]

    # --- 3D Plotting ---
    ax.clear()
    
    # Set figure background color to black
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Set axes colors to white
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')
    
    # Convert occupancy grid to 3D visualization
    grid_width = planner.occ_grid.info.width
    grid_height = planner.occ_grid.info.height
    grid_data = np.array(planner.occ_grid.data).reshape((grid_height, grid_width))
    
    # Create coordinate grids for the surface plot
    x = np.linspace(origin.x, origin.x + width * res, grid_width)
    y = np.linspace(origin.y, origin.y + height * res, grid_height)
    X, Y = np.meshgrid(x, y)
    
    # Create a flat base terrain with minor height variations for free/unknown space
    Z = np.zeros((grid_height, grid_width))
    
    # Set heights based on occupancy values - just for free/unknown space
    for i in range(grid_height):
        for j in range(grid_width):
            if grid_data[i, j] < 0:  # Free space (negative values)
                # Very slight variation in height for free space
                Z[i, j] = 0.01 * (abs(grid_data[i, j]) / 128.0)
            elif grid_data[i, j] < 127:  # Unknown space
                Z[i, j] = 0.02
            else:  # Obstacles (will be drawn separately)
                Z[i, j] = 0.0
    
    # Normalize grid data for coloring the base terrain
    norm_data = (grid_data + 128) / 255.0
    
    # Create a custom colormap: black for free space, gray for unknown
    cmap = colors.LinearSegmentedColormap.from_list(
        'OccupancyMap', [(0.0, 'white'), (0.5, 'black'), (1.0, 'black')], N=256)
    
    # Plot the surface using the custom colormap (base terrain)
    surf = ax.plot_surface(X, Y, Z, 
                          facecolors=cmap(norm_data),
                          linewidth=0, 
                          antialiased=False, 
                          alpha=1.0,
                          zorder=-1)
    
    # Add 3D prisms for obstacles
    obstacle_height = 0.8  # Height of obstacle prisms
    dx = dy = res  # Width and depth of each prism (cell size)
    
    # Collect coordinates for all obstacles
    obstacle_xs = []
    obstacle_ys = []
    obstacle_zs = []
    
    for i in range(grid_height):
        for j in range(grid_width):
            if grid_data[i, j] == 127:  # Obstacle cell
                cell_x = origin.x + j * res
                cell_y = origin.y + i * res
                obstacle_xs.append(cell_x)
                obstacle_ys.append(cell_y)
                obstacle_zs.append(0)  # Start from ground level
    
    # Draw obstacle prisms using bar3d
    if obstacle_xs:  # Only draw if there are obstacles
        ax.bar3d(
            obstacle_xs, obstacle_ys, obstacle_zs,
            dx * np.ones(len(obstacle_xs)),  # width
            dy * np.ones(len(obstacle_xs)),  # depth
            obstacle_height * np.ones(len(obstacle_xs)),  # height
            color='red',
            alpha=1.0,
            shade=True,
            zsort='max'
        )

    
    # Visualize FOV rays from the drone
    ray_height = 0.1  # Slightly above the ground
    yaw = planner.get_yaw(current_pose.pose.orientation)
    half_fov = planner.fov / 2.0
    num_rays = 10  # Number of rays to visualize
    
    # Draw current FOV rays
    for i in range(num_rays + 1):
        angle_deg = -half_fov + i * (planner.fov / num_rays)
        angle_rad = math.radians(angle_deg) + yaw
        ray_end_x = current_pose.pose.position.x + planner.max_range * math.cos(angle_rad)
        ray_end_y = current_pose.pose.position.y + planner.max_range * math.sin(angle_rad)
        
        # Check for intersection with obstacles using Bresenham
        grid_x = int((current_pose.pose.position.x - origin.x) / res)
        grid_y = int((current_pose.pose.position.y - origin.y) / res)
        end_grid_x = int((ray_end_x - origin.x) / res)
        end_grid_y = int((ray_end_y - origin.y) / res)
        ray_cells = planner.bresenham_line(grid_x, grid_y, end_grid_x, end_grid_y)
        
        # Find the first obstacle (if any)
        ray_length = planner.max_range
        for idx, (cell_x, cell_y) in enumerate(ray_cells):
            if 0 <= cell_x < width and 0 <= cell_y < height:
                cell_idx = cell_y * width + cell_x
                if grid[cell_idx] == 127:  # Found obstacle
                    # Calculate the actual distance to this obstacle
                    cell_world_x = origin.x + (cell_x + 0.5) * res
                    cell_world_y = origin.y + (cell_y + 0.5) * res
                    ray_length = math.hypot(
                        cell_world_x - current_pose.pose.position.x,
                        cell_world_y - current_pose.pose.position.y
                    )
                    break
        
        # Draw ray from drone to obstacle or max range
        new_end_x = current_pose.pose.position.x + ray_length * math.cos(angle_rad)
        new_end_y = current_pose.pose.position.y + ray_length * math.sin(angle_rad)
        if num_visited < 4:
            # Draw the ray
            ax.plot(
                [current_pose.pose.position.x, new_end_x], 
                [current_pose.pose.position.y, new_end_y], 
                [ray_height, ray_height], 
                color='yellow', 
                alpha=0.4,
                linewidth=1.0,
                zorder=10,
            )
        
        # Plot planned path as elevated line
        if path_coords is not None:
            xs, ys = zip(*path_coords)
            zs = [0.1] * len(xs)  # Slightly above the ground
            ax.plot(xs, ys, zs, linestyle='-', color='cyan', linewidth=2, alpha=0.7)
        
        # Highlight the current and next positions
        ax.scatter([current_pose.pose.position.x], [current_pose.pose.position.y], [0.1], 
                marker='o', s=100, color='yellow', edgecolors='white', linewidth=1)
        if next_pose is not None and next_pose is not current_pose:
            ax.scatter([next_pose.pose.position.x], [next_pose.pose.position.y], [0.1], 
                    marker='o', s=100, color='cyan', edgecolors='white', linewidth=1)
    
    # Add camera indicators
    # add_camera_indicator_3d(ax, current_pose, color='yellow', size=0.4)
    # if next_pose is not None and next_pose is not current_pose:
    #     add_camera_indicator_3d(ax, next_pose, color='cyan', size=0.4)
    
    # Plot 1 with 3D markers
    target_height = 0.1  # Higher than path for visibility
    for target in targets:
        if getattr(target, "visited", False):
            ax.scatter([target.pose.position.x], [target.pose.position.y], [target_height], 
                       marker='o', s=120, color='cyan', edgecolors='white', linewidth=1, zorder=10,
)
        else:
            ax.scatter([target.pose.position.x], [target.pose.position.y], [target_height], 
                       marker='x', s=120, color='lime', linewidth=2, zorder=10)
    
    # Add grid lines
    x_min, x_max = origin.x, origin.x + width * res
    y_min, y_max = origin.y, origin.y + height * res
    ax.grid(False)
    # Draw grid lines (spaced every 0.5m)
    grid_step = 0.1
    for x in np.arange(np.floor(x_min), np.ceil(x_max) + grid_step, grid_step):
        ax.plot([x, x], [y_min, y_max], [0, 0], color='white', alpha=0.2, linewidth=0.5)
    for y in np.arange(np.floor(y_min), np.ceil(y_max) + grid_step, grid_step):
        ax.plot([x_min, x_max], [y, y], [0, 0], color='white', alpha=0.2, linewidth=0.5)
    grid_step = 0.5
    for x in np.arange(np.floor(x_min), np.ceil(x_max) + grid_step, grid_step):
        ax.plot([x, x], [y_min, y_max], [0, 0], color='white', alpha=0.6, linewidth=1)
    for y in np.arange(np.floor(y_min), np.ceil(y_max) + grid_step, grid_step):
        ax.plot([x_min, x_max], [y, y], [0, 0], color='white', alpha=0.6, linewidth=1)

    
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.line.set_lw(0.)
    ax.set_zticks([])
    # Set plot limits and labels
    ax.set_xlim([origin.x, origin.x + width * res])
    ax.set_ylim([origin.y, origin.y + height * res])
    ax.set_zlim([0, 1.0])
    # ax.set_xlabel('X (meters)')
    # ax.set_ylabel('Y (meters)')
    # ax.set_zlabel('Z (meters)')
    # ax.set_title('3D Occupancy Grid Map', color='white')
    
    # Set equal aspect ratio for the 3D plot
    ax.set_box_aspect([1, 1, 0.3])
    ax.view_init(elev=30, azim=cur_Frame+1)

    # Update the figure
    # fig.canvas.draw()
    
    current_pose = next_pose
# --- Keypress event handler ---
def on_key(event):
    if event.key == "right":  # Advance planner
        update_plot()
    elif event.key == "escape":  # Exit simulation
        plt.close()
import matplotlib.animation as animation
cur_Frame = 0
def animate_rotation(frame):
    global cur_Frame
    cur_Frame = frame
    """Rotate the 3D view for animation"""
    ax.view_init(elev=30, azim=frame)
    return []

# --- Main function ---
def main():
    global planner, current_pose, fig, ax, targets, use_prm_planner

    # Set this flag to True to use PRM planning; otherwise, A* is used.
    use_prm_planner = False  # Change to True if you want to use PRM

    # Create planner instance.
    planner = Planner()

    # Initialize the drone's starting pose.
    current_pose = PoseStamped()
    current_pose.pose.position.x = 0.0
    current_pose.pose.position.y = 0.0
    # Set the initial yaw so that the camera's FOV is correctly oriented.
    yaw = math.radians(45)
    current_pose.pose.orientation.x = 0.0
    current_pose.pose.orientation.y = 0.0
    current_pose.pose.orientation.z = math.sin(yaw / 2)
    current_pose.pose.orientation.w = math.cos(yaw / 2)
    
    # Initialize the drone's starting pose.
    current_pose = PoseStamped()
    current_pose.pose.position.x = 0.0
    current_pose.pose.position.y = 0.0
    # Set the initial yaw so that the camera's FOV is correctly oriented.
    yaw = math.radians(45)
    current_pose.pose.orientation.x = 0.0
    current_pose.pose.orientation.y = 0.0
    current_pose.pose.orientation.z = math.sin(yaw / 2)
    current_pose.pose.orientation.w = math.cos(yaw / 2)

    # Add fixed obstacles.
    obs1 = PoseStamped()
    obs1.pose.position.x = 1.0
    obs1.pose.position.y = 0.5
    planner.add_obstacle(current_pose, obs1)
    
    obs2 = PoseStamped()
    obs2.pose.position.x = -1.0
    obs2.pose.position.y = -0.5
    planner.add_obstacle(current_pose, obs2)

    obs3 = PoseStamped()
    obs3.pose.position.x = -1.0
    obs3.pose.position.y = 0.5
    planner.add_obstacle(current_pose, obs3)

    # Drop 5 targets manually with some placed behind obstacles.
    targets.clear()
    # Manually defined target positions (x, y):
    # - Targets (0.8, 0.8) and (0.5, 1.5) are likely visible.
    # - Targets (1.5, 0.5), (-1.5, -0.5) and (-1.5, 0.8) are placed behind obstacles.
    manual_targets = [
        # (0.8, 0.8),    # visible in front
        (1.5, 0.5),    # behind obstacle near (1.0, 0.5)
        (-1.5, -0.5),  # behind obstacle near (-1.0, -0.5)
        (0.5, 1.5),    # visible in front
        (-1.5, 0.8)    # behind obstacle near (-1.0, 0.5)
    ]

    for i, (tx, ty) in enumerate(manual_targets):
        target_pose = PoseStamped()
        target_pose.header.frame_id = "map"
        target_pose.pose.position.x = tx
        target_pose.pose.position.y = ty
        target_pose.pose.position.z = 0.0
        target_yaw = 0.0
        target_pose.pose.orientation.x = 0.0
        target_pose.pose.orientation.y = 0.0
        target_pose.pose.orientation.z = math.sin(target_yaw / 2)
        target_pose.pose.orientation.w = math.cos(target_yaw / 2)
        target_pose.visited = False  # Mark as unvisited.
        targets.append(target_pose)
        print(f"Dropped target {i+1} at ({tx:.2f}, {ty:.2f}).")


    
    # Setup 3D Matplotlib figure with black background
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    
    # Set up key press event handler
    fig.canvas.mpl_connect("key_press_event", on_key)
    update_plot()
    ani = animation.FuncAnimation(
        fig, animate_rotation, 
        frames=np.arange(0, 360, 0.1), 
        interval=10,                    
        blit=True,
        repeat=True
    )

    plt.show()
    
if __name__ == "__main__":
    main()
