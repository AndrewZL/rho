from setuptools import setup

package_name = "rho"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/mavros_launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jetson",
    maintainer_email="jetson@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "comm = rho.comm:main",
            "control = rho.control:main",
            "pose_realsense = rho.pose_realsense:main",
        ],
    },
)
