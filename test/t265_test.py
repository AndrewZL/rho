import cv2
import matplotlib.pyplot as plt
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class FisheyeSubscriber(Node):
    def __init__(self):
        super(FisheyeSubscriber, self).__init__('fisheye_subscriber')
        
        self.bridge = CvBridge()
        
        self.subscription_left = self.create_subscription(
            Image,
            '/camera/fisheye1/image_raw',
            self.image_callback_left,
            10)
        
        self.subscription_right = self.create_subscription(
            Image,
            '/camera/fisheye2/image_raw',
            self.image_callback_right,
            10)
        
        self.left_image = None
        self.right_image = None

    def image_callback_left(self, msg):
        try:
            self.left_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.display_images()
        except Exception as e:
            self.get_logger().error('Error processing left image: ' + str(e))

    def image_callback_right(self, msg):
        try:
            self.right_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.display_images()
        except Exception as e:
            self.get_logger().error('Error processing right image: ' + str(e))

    def display_images(self):
        if self.left_image is not None and self.right_image is not None:
            stacked_image = cv2.hconcat([self.left_image, self.right_image])
            plt.imshow('Fisheye Stereo Images', stacked_image)
            plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = FisheyeSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

