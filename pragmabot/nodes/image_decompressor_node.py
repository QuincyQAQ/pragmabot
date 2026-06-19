#!/usr/bin/env python3
"""ROS node that decompresses CompressedImage messages to raw Image messages."""

import numpy as np
import cv2

import rospy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge


class ImageDecompressor:
    """Decompress ROS CompressedImage messages and republish as raw Image messages."""

    def __init__(self) -> None:
        """Initialize the decompressor node with a subscriber and publisher."""
        rospy.init_node("image_decompressor_node", anonymous=True)
        self.bridge = CvBridge()

        # Subscribe to fixed internal topic name — will be remapped via launch file
        self.sub = rospy.Subscriber("input/compressed", CompressedImage, self.callback, queue_size=1)

        # Publish to fixed internal topic name — will be remapped via launch file
        self.pub = rospy.Publisher("output/image", Image, queue_size=1)

    def run(self) -> None:
        """Block on rospy.spin() until shutdown."""
        rospy.spin()

    def callback(self, msg: CompressedImage) -> None:
        """Decompress a single CompressedImage and publish as a raw Image."""
        try:
            # Decompress image
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if cv_image is None:
                rospy.logwarn("Failed to decode compressed image")
                return

            # Convert to ROS Image message
            image_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            image_msg.header = msg.header  # preserve time and frame_id

            # Publish
            self.pub.publish(image_msg)

        except Exception as e:
            rospy.logerr(f"Error in decompression: {e}")


if __name__ == "__main__":
    try:
        decompressor = ImageDecompressor()
        decompressor.run()
    except rospy.ROSInterruptException:
        pass
