from typing import Dict, List, Optional

import apriltag
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def scan_apriltag(img: Image, tag_size: float = 0.139) -> Optional[List[Dict[str, np.ndarray]]]:
    """
    Scan for AprilTags in the given image.
    """

    # Camera parameters
    camera_matrix = np.array([[285.5625915527344, 0.0, 424.74578857421875], [0.0, 285.5155029296875, 403.2914123535156], [0.0, 0.0, 1.0]], dtype=np.float32)
    dist_coeffs = np.array([-0.008453547954559326, 0.045836448669433594, -0.043336059898138046, 0.008145838975906372], dtype=np.float32)

    # Convert the ROS image to an OpenCV image.
    while img == None:
        continue
    try:
        cv_image = CvBridge().imgmsg_to_cv2(img, desired_encoding="bgr8")
    except Exception as e:
        print(f"CV Bridge conversion failed: {e}")
        return None

    DIM = (848, 800)  # width, height from your camera info.
    undistorted_image = cv2.fisheye.undistortImage(cv_image, camera_matrix, D=dist_coeffs, Knew=camera_matrix)

    # Convert undistorted image to grayscale.
    gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
    # Preprocessing: Apply Contrast Limited Adaptive Histogram Equalization for better contrast and add sharpness
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 1.0)
    amount = 1.5
    final = np.clip((1 + amount) * enhanced - amount * blur, 0, 255).astype(np.uint8)

    # Initialize the AprilTag detector.
    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)
    detections = detector.detect(final)

    detection_array = []

    # Process each detected tag.
    for det in detections:
        # Define 3D object points for the tag corners (centered at 0,0,0).
        object_points = np.array(
            [[-tag_size / 2, -tag_size / 2, 0], [tag_size / 2, -tag_size / 2, 0], [tag_size / 2, tag_size / 2, 0], [-tag_size / 2, tag_size / 2, 0]], dtype=np.float32
        )
        # The detected corners are in image coordinates.
        image_points = np.array(det.corners, dtype=np.float32)
        # Use solvePnP to compute the translation vector.
        ret, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if ret:
            detection_array.append({"detection": det, "rvec": rvec, "tvec": tvec})

    return detection_array
