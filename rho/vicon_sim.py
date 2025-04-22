import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy


class ViconSim(Node):
    """
    A ROS2 node that simulates Vicon data by publishing a modified version of the MAVROS local position data.
    """

    def __init__(self):
        super().__init__("vicon_sim")
        qos_profile = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        self.create_subscription(PoseStamped, "/mavros/local_position/pose", self.odom_callback, qos_profile)
        self.vicon_pub = self.create_publisher(PoseStamped, "/vicon/ROB498_Drone/ROB498_Drone", qos_profile)

    def odom_callback(self, msg: PoseStamped):
        msg.pose.position.x = 5.0
        msg.pose.position.y = 5.0
        msg.pose.position.z = 0.0

        self.vicon_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ViconSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
