import numpy as np
from geometry_msgs.msg import Quaternion


### Math Utils
def quaternion_to_rotation_matrix(q: list) -> np.ndarray:
    """
    Convert a quaternion [x, y, z, w] to a rotation matrix.
    """
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def quaternion_to_euler(q: list) -> tuple:
    """
    Convert a quaternion [x, y, z, w] to Euler angles (roll, pitch, yaw) in radians.
    """
    x, y, z, w = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles (in radians) to a rotation matrix.
    """
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

    return R_z @ R_y @ R_x


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles (in radians) to a quaternion (x, y, z, w).
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    q = np.array([sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy])  # x  # y  # z  # w
    return q


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions q1 and q2.
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,  # x
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,  # y
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,  # z
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,  # w
        ]
    )


def get_yaw(orientation: Quaternion) -> float:
    """
    Return yaw from a quaternion orientation.
    """
    _, _, yaw = quaternion_to_euler([orientation.x, orientation.y, orientation.z, orientation.w])
    return yaw


### Planning Utils
def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> list:
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


def shannon_entropy(p: float) -> float:
    """Compute Shannon's entropy (in bits) for a probability p."""
    if p == 0 or p == 1:
        return 0
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
