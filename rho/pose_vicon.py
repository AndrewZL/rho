# pose_vicon.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from rclpy.qos import qos_profile_system_default


class PoseVicon(Node):
    """
    This node subscribes to the Vicon pose topic and publishes the pose to the MAVROS vision pose topic.
    """

    def __init__(self):
        super().__init__("pose_vicon")

        self.create_subscription(PoseStamped, "/vicon/ROB498_Drone/ROB498_Drone", self.vicon_callback, qos_profile_system_default)
        self.vision_pose_pub = self.create_publisher(PoseStamped, "/mavros/vision_pose/pose", qos_profile_system_default)

    def vicon_callback(self, msg: PoseStamped):
        vision_pose = PoseStamped()
        vision_pose.header.frame_id = "map"
        vision_pose.header.stamp = self.get_clock().now().to_msg()

        vision_pose.pose.position = msg.pose.position
        vision_pose.pose.orientation = msg.pose.orientation
        self.vision_pose_pub.publish(vision_pose)


def main(args=None):
    rclpy.init(args=args)
    node = PoseVicon()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
