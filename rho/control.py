import copy
import math
import threading
import time

import rclpy
import serial
from geometry_msgs.msg import PoseStamped
from mavros_msgs.srv import CommandBool, SetMode
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from sensor_msgs.msg import Image

from rho.apriltag_realsense import scan_apriltag
from rho.planner import Planner
from rho.utils import euler_to_quaternion


class Controller(Node):
    """
    This node controls the drone's flight path and communication with the Arduino.
    It subscribes to the MAVROS vision pose topic and publishes the desired pose to the MAVROS setpoint position topic.
    It also handles the communication with the Arduino for dropping aid.
    """

    def __init__(self):
        super().__init__("controller")

        # Control
        self.pose_sub = self.create_subscription(PoseStamped, "/mavros/vision_pose/pose", self.pose_callback, qos_profile_system_default)
        self.local_pose_pub = self.create_publisher(PoseStamped, "/mavros/setpoint_position/local", qos_profile_system_default)
        self.arming_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.mode_client = self.create_client(SetMode, "/mavros/set_mode")

        self.image_msg = None
        self.subscription = self.create_subscription(Image, "/camera/fisheye1/image_raw", self.image_callback, qos_profile_system_default)

        # Internal state
        self.pose_curr = None
        self.pose_init = None
        self.pose_des = PoseStamped()
        self.pose_des.header.frame_id = "map"

        self.mode = "IDLE"  # Modes: IDLE, ASCENDING, TEST, LANDING, LANDED
        self.altitude_threshold = 0.01  # tolerance for altitude error
        self.delta_z_des = 0.4  # operational altitude

        # Control Loop
        self.timer = self.create_timer(0.05, self.timer_callback)

        # Planning
        self.planner = Planner(self.get_logger())
        self.targets_saved = 0
        self.num_targets = 3
        self.cur_waypoint = 0
        self.waypoint_radius = 0.1
        self.is_target = None

        # Servo
        port = "/dev/ttyUSB1"
        self.ser = None
        self.ser = serial.Serial(port, 115200, timeout=0.1)
        if not self.ser.is_open:
            raise serial.SerialException("[START] Port exists but is not open.")
        self.get_logger().info("[START] Arduino connected successfully on port {}".format(port))
        self.lock = threading.Lock()
        self.waiting_for_confirm = False
        self.last_command = None
        if self.ser:
            self.confirm_thread = threading.Thread(target=self._listen_for_confirmations)
            self.confirm_thread.daemon = True
            self.confirm_thread.start()

    ### SERVO ###
    def _listen_for_confirmations(self):
        """Background thread to read serial and check for confirmations."""
        while self.ser and self.ser.is_open:
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode().strip()
                    if line == "{}_CONFIRM".format(self.last_command):
                        with self.lock:
                            self.waiting_for_confirm = False
                            self.last_command = None
            except serial.SerialException:
                self.get_logger().info("Error: Lost connection to Arduino.")
                self.ser = None  # Mark as disconnected
                break
            time.sleep(0.05)

    def send_command(self, command: str) -> bool:
        """Send a command only if no other command is pending confirmation."""
        if not self.is_connected():
            self.get_logger().info("Error: Arduino is not connected.")
            return False

        with self.lock:
            if self.waiting_for_confirm:
                self.get_logger().info("Waiting for confirmation before sending another command.")
                return False
            self.last_command = command
            self.waiting_for_confirm = True

        full_command = "{}\n".format(command)
        try:
            self.ser.write(full_command.encode())
            self.get_logger().info("[Jetson to Arduino] Sent command: {}".format(command))
            return True
        except serial.SerialException:
            self.get_logger().info("Error: Failed to send command.")
            self.ser = None  # Mark as disconnected
            return False

    def is_connected(self):
        """Check if the Arduino is connected."""
        return self.ser is not None and self.ser.is_open

    def close(self):
        if self.ser:
            self.ser.close()
            self.get_logger().info("Serial connection closed.")
            self.ser = None

    ### CALLBACKS ###
    def pose_callback(self, msg: PoseStamped):
        self.pose_curr = msg

    def image_callback(self, msg: Image):
        self.image_msg = msg

    ### COMMAND HANDLERS ###
    def handle_launch(self):
        if self.pose_curr is None:
            self.get_logger().error("Current pose is not available. Cannot launch!")
            return
        self.get_logger().info("LAUNCH command received. Attempting to arm drone.")

        self.set_initial_pose()

        self.arm()
        self.mode = "ASCENDING"
        self.get_logger().info("[MODE] Ascending")

    def handle_test(self, command: str):
        if self.mode not in ["ASCENDING"]:
            self.get_logger().warn("Drone must be hovering to start test.")
            return
        if not self.waypoints:
            self.get_logger().warn("No target waypoints! Can't test")
            return
        self.mode = "TEST"
        self.get_logger().info("[MODE] Testing")

    def handle_land(self):
        self.set_auto_land_mode()
        self.mode = "LANDING"
        self.get_logger().info("[MODE] Landing")

    def handle_abort(self):
        self.get_logger().info("ABORT command received. Switching to AUTO.LAND mode immediately!")
        self.set_auto_land_mode()
        self.mode = "LANDING"

    def set_auto_land_mode(self):
        if not self.mode_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("SetMode service is not available!")
            return
        req = SetMode.Request()
        req.custom_mode = "AUTO.LAND"
        future = self.mode_client.call_async(req)
        future.add_done_callback(lambda f: self.get_logger().info("AUTO.LAND mode activated."))

    def arm(self):
        if not self.arming_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Arming service is not available!")
            return
        req = CommandBool.Request()
        req.value = True
        future = self.arming_client.call_async(req)
        self.get_logger().info("waiting for arm")
        future.add_done_callback(lambda f: self.get_logger().info("Drone arm message sent."))
        if future.result() is not None and future.result().success:
            self.get_logger().info("armed")
        else:
            self.get_logger().warn("can't arm")

    def disarm(self):
        if not self.arming_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("Arming service is not available!")
            return
        req = CommandBool.Request()
        req.value = False
        future = self.arming_client.call_async(req)
        future.add_done_callback(lambda f: self.get_logger().info("Drone disarmed."))

    def set_initial_pose(self):
        self.pose_init = copy.deepcopy(self.pose_curr)
        self.pose_des.pose.position.x = self.pose_init.pose.position.x
        self.pose_des.pose.position.y = self.pose_init.pose.position.y
        self.pose_des.pose.position.z = self.delta_z_des + 0.2
        self.pose_des.pose.orientation = self.pose_init.pose.orientation
        self.waypoints = [[self.pose_des.pose.position.x, self.pose_des.pose.position.y, self.delta_z_des + 0.2, euler_to_quaternion(0, 0, 0)]]

    def check_target(self) -> bool:
        return (
            math.sqrt(
                (self.pose_curr.pose.position.x - self.waypoints[self.cur_waypoint][0]) ** 2
                + (self.pose_curr.pose.position.y - self.waypoints[self.cur_waypoint][1]) ** 2
            )
            < self.waypoint_radius
        )

    def add_last_waypoint(self):
        self.waypoints.append([0.0, 0.0, self.delta_z_des, euler_to_quaternion(0, 0, 0)])
        self.get_logger().info(f"[PLANNER] Added last waypoint: {self.waypoints[-1]}")

    def global_planner(self):
        self.get_logger().info("[PLANNER] Target Reached")

        # reset the planner
        self.cur_waypoint = 0
        self.waypoints.clear()

        # check if done task
        if self.is_target:
            self.get_logger().info("finished a target")
            self.targets_saved += 1
            # drop aid to target
            if self.send_command("DROP_X"):
                self.get_logger().info("[SERVO] Command Sent")
            else:
                self.get_logger().info("[SERVO] Command Failed")

            if self.targets_saved >= self.num_targets:
                self.get_logger().info("[PLANNER] All targets reached")
                path_coords = self.planner.go_to_home(self.pose_curr)
                self.mode = "RETURNING"
                self.get_logger().info(f"Next Waypoint {path_coords[-1]}, {self.is_target}")
                # Failure: hopefully this never happens
                if path_coords is None:
                    self.get_logger().info("[PLANNER] No path found")
                    return
                for i in range(len(path_coords)):
                    self.waypoints.append([path_coords[i][0], path_coords[i][1], self.delta_z_des, *euler_to_quaternion(0, 0, path_coords[i][2])])
                return
        # OPERATION LOOP
        # 1. detect tags
        detections = scan_apriltag(self.image_msg)
        self.pose_des.header.stamp = self.get_clock().now().to_msg()
        self.local_pose_pub.publish(self.pose_des)

        # 2. add to planner
        obstacles_found = self.planner.process_detections(self.pose_curr, detections)
        if not obstacles_found:
            self.set_auto_land_mode()
            print("not obs find")
            return
        self.pose_des.header.stamp = self.get_clock().now().to_msg()
        self.local_pose_pub.publish(self.pose_des)

        # 3. plan path
        path_coords, self.is_target = self.planner.get_next_poses(self.pose_curr, self.local_pose_pub, self.pose_des)
        self.get_logger().info(f"Next Waypoint {path_coords[-1]}, {self.is_target}")
        # Failure: hopefully this never happens
        if path_coords is None:
            self.get_logger().info("[PLANNER] No path found")
            return
        # 5. send path to drone
        for i in range(len(path_coords)):
            self.waypoints.append([path_coords[i][0], path_coords[i][1], self.delta_z_des, *euler_to_quaternion(0, 0, path_coords[i][2])])

    ### CONTROL LOOP ###
    def timer_callback(self):
        # Publish the desired pose
        if self.pose_des is not None and self.mode != "LANDING":
            self.pose_des.header.stamp = self.get_clock().now().to_msg()
            self.local_pose_pub.publish(self.pose_des)

        # ABORT if SLAM error
        if self.pose_curr is not None and self.pose_curr.pose.position.z < -0.5:
            self.get_logger().info("SLAM ERROR, ABORTING")
            self.mode == "LANDING"
            self.set_auto_land_mode()

        # Handle Land
        elif self.mode == "LANDING":
            self.get_logger().info("[MODE] Landing")
            if self.pose_curr is not None and self.pose_init is not None:
                current_z = self.pose_curr.pose.position.z
                landing_z = self.pose_init.pose.position.z
                if abs(current_z - landing_z) < self.altitude_threshold:
                    self.get_logger().info("[MODE] Landing Complete")
                    self.disarm()

        # Main Control Loop
        elif self.mode == "TEST":
            # if reached waypoint, do updates
            if self.check_target():
                self.get_logger().info("next waypoint")
                self.cur_waypoint += 1
                if self.cur_waypoint == len(self.waypoints):
                    self.global_planner()
                self.pose_des.pose.position.x = self.waypoints[self.cur_waypoint][0]
                self.pose_des.pose.position.y = self.waypoints[self.cur_waypoint][1]
                self.pose_des.pose.position.z = self.waypoints[self.cur_waypoint][2]
                self.pose_des.pose.orientation.x = self.waypoints[self.cur_waypoint][3]
                self.pose_des.pose.orientation.y = self.waypoints[self.cur_waypoint][4]
                self.pose_des.pose.orientation.z = self.waypoints[self.cur_waypoint][5]
                self.pose_des.pose.orientation.w = self.waypoints[self.cur_waypoint][6]
            # otherwise, keep moving towards the target
        elif self.mode == "RETURNING":
            if self.check_target():
                self.cur_waypoint += 1
                if self.cur_waypoint == len(self.waypoints):
                    self.mode = "LANDING"
                    self.get_logger().info("[MODE] Landing")
                    self.set_auto_land_mode()
                self.pose_des.pose.position.x = self.waypoints[self.cur_waypoint][0]
                self.pose_des.pose.position.y = self.waypoints[self.cur_waypoint][1]
                self.pose_des.pose.position.z = self.waypoints[self.cur_waypoint][2]
                self.pose_des.pose.orientation.x = self.waypoints[self.cur_waypoint][3]
                self.pose_des.pose.orientation.y = self.waypoints[self.cur_waypoint][4]
                self.pose_des.pose.orientation.z = self.waypoints[self.cur_waypoint][5]
                self.pose_des.pose.orientation.w = self.waypoints[self.cur_waypoint][6]


def main(args=None):
    rclpy.init(args=args)
    try:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
