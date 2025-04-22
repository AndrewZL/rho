import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from rho.control import Controller

controller = None


# Callback handlers that call the corresponding methods on the controller
def handle_launch():
    print("Launch Requested. Your drone should take off.")
    controller.handle_launch()


def handle_test():
    print("Test Requested. Your drone should perform the required tasks. Recording starts now.")
    controller.handle_test("TEST")


def handle_land():
    print("Land Requested. Your drone should land.")
    controller.handle_land()


def handle_abort():
    print("Abort Requested. Your drone should land immediately due to safety considerations")
    controller.handle_abort()


# Service callbacks
def callback_launch(request, response):
    handle_launch()
    return response


def callback_test(request, response):
    handle_test()
    return response


def callback_land(request, response):
    handle_land()
    return response


def callback_abort(request, response):
    handle_abort()
    return response


class CommNode(Node):
    def __init__(self):
        super().__init__("rob498_drone_02")
        self.srv_launch = self.create_service(Trigger, "rob498_drone_2/comm/launch", callback_launch)
        self.srv_test = self.create_service(Trigger, "rob498_drone_2/comm/test", callback_test)
        self.srv_land = self.create_service(Trigger, "rob498_drone_2/comm/land", callback_land)
        self.srv_abort = self.create_service(Trigger, "rob498_drone_2/comm/abort", callback_abort)


def main(args=None):
    rclpy.init(args=args)
    global controller
    controller = Controller()
    comm_node = CommNode()

    # Use a multi-threaded executor to run both the controller and comm node concurrently
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(controller)
    executor.add_node(comm_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        comm_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
