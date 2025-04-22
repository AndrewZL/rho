from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="mavros",
                executable="mavros_node",
                parameters=["/home/jetson/ws/src/rho/config/mavros_param.yaml"],
                output="screen",
            ),
            Node(
                package="rho",
                executable="comm",
                name="comm",
                output="screen",
            ),
            Node(
                package="rho",
                executable="pose_realsense",
                name="pose_realsense",
                output="screen",
            ),
        ]
    )
