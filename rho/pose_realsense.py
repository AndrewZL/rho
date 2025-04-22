import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default

from rob498.utils import *


class PoseRealSense(Node):
    """
    This node subscribes to the Intel RealSense T265 pose topic and publishes the pose to the MAVROS vision pose topic.
    The pose is transformed from the camera frame to the vicon frame assuming the drone faces the back wall on initialization.
    """

    def __init__(self):
        super().__init__("pose_realsense")

        # Subscribe to the Intel RealSense T265 pose topic
        self.create_subscription(Odometry, "/camera/pose/sample", self.realsense_callback, qos_profile_system_default)  # T265 topic

        # Publisher for MAVROS vision pose
        self.vision_pose_pub = self.create_publisher(PoseStamped, "/mavros/vision_pose/pose", qos_profile_system_default)

    def transform_pose(self, msg: Odometry) -> PoseStamped:
        """
        Transform T265 pose from camera frame to ENU frame.
        """
        vision_pose = PoseStamped()
        vision_pose.header.frame_id = "map"  # Set correct frame ID
        vision_pose.header.stamp = self.get_clock().now().to_msg()

        # Convert orientation using quaternion transformation
        q_orig = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])

        # negative 90-degree yaw rotation
        q_rot = euler_to_quaternion(0, 0, -np.pi / 2)
        q_new = quaternion_multiply(q_rot, q_orig)

        vision_pose.pose.orientation.x = q_new[0]
        vision_pose.pose.orientation.y = q_new[1]
        vision_pose.pose.orientation.z = q_new[2]
        vision_pose.pose.orientation.w = q_new[3]

        # Transform position
        p_orig = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        p_new = R @ p_orig

        vision_pose.pose.position.x = p_new[0]
        vision_pose.pose.position.y = p_new[1]
        vision_pose.pose.position.z = p_new[2]

        return vision_pose

    def realsense_callback(self, msg):
        transformed_pose = self.transform_pose(msg)
        self.vision_pose_pub.publish(transformed_pose)


def main(args=None):
    rclpy.init(args=args)
    node = PoseRealSense()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
