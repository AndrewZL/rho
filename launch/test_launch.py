from launch_ros.actions import Node

from launch import LaunchDescription


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="rho",  # your package name
                executable="vicon_sim",  # the python node file
                name="vicon_sim",
                output="screen",
            ),
            Node(
                package="rho",  # your package name
                executable="testing_task2",  # the python node file
                name="testing_task2",
                output="screen",
            ),
        ]
    )
