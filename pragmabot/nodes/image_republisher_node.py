#!/usr/bin/env python3
"""ROS node that republishes depth/RGB images on an external trigger."""

import threading
import os

import cv2
import numpy as np

import rospy
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Point


class ImageRepublisher:
    """Republish depth/RGB images on an external mouse-click trigger."""

    def __init__(self) -> None:
        """Initialize the republisher node with trigger-gated image forwarding."""
        rospy.init_node("image_republisher_node", anonymous=True)

        # Parameters
        self._max_count = rospy.get_param("~max_count", 1)
        self._save_image = rospy.get_param("~save_image", False)
        self._save_folder = rospy.get_param("~save_folder", "/tmp")

        # State
        self._rgb_remaining = 0
        self._depth_remaining = 0

        # Separate locks so slow RGB disk I/O doesn't block Depth publishing
        self._rgb_lock = threading.Lock()
        self._depth_lock = threading.Lock()

        # Publishers
        self._depth_pub = rospy.Publisher("output/depth", Image, queue_size=1)
        self._rgb_pub = rospy.Publisher("output/rgb", CompressedImage, queue_size=1)

        # Subscribers
        self._depth_sub = rospy.Subscriber("input/depth", Image, self.depth_callback)
        self._rgb_sub = rospy.Subscriber("input/rgb", CompressedImage, self.rgb_callback)

        # Trigger subscriber
        self._click_sub = rospy.Subscriber("image_click", Point, self.trigger_callback)

    def run(self) -> None:
        """Block on rospy.spin() until shutdown."""
        rospy.spin()

    def trigger_callback(self, msg: Point) -> None:
        """Handle a mouse-click trigger to queue images for republication."""
        rospy.loginfo("Mouse trigger received! Queuing images for republication...")

        # Safely open the gates
        with self._rgb_lock:
            self._rgb_remaining = self._max_count

        with self._depth_lock:
            self._depth_remaining = self._max_count

    def depth_callback(self, msg: Image) -> None:
        """Forward a depth image if a trigger is pending."""
        with self._depth_lock:
            if self._depth_remaining > 0:
                self._depth_pub.publish(msg)

                # Calculate current count for logging (1 to max_count)
                current_count = self._max_count - self._depth_remaining + 1
                rospy.loginfo(f"Published depth image {current_count}/{self._max_count}")

                self._depth_remaining -= 1

    def rgb_callback(self, msg: CompressedImage) -> None:
        """Forward an RGB image if a trigger is pending, optionally saving to disk."""
        should_save = False

        with self._rgb_lock:
            if self._rgb_remaining > 0:
                self._rgb_pub.publish(msg)

                # Calculate current count for logging (1 to max_count)
                current_count = self._max_count - self._rgb_remaining + 1
                rospy.loginfo(f"Published RGB image {current_count}/{self._max_count}")

                # Flag for saving, but wait to execute it
                if self._save_image and current_count == 1:
                    should_save = True

                self._rgb_remaining -= 1

        # Perform the slower disk I/O completely outside the lock
        # so we don't block the next RGB or Depth frames from being processed
        if should_save:
            self._save_rgb_image(msg)

    def _save_rgb_image(self, msg: CompressedImage) -> None:
        """Decode and save a compressed RGB image to disk."""
        # Convert CompressedImage to OpenCV image
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if cv_image is None:
            rospy.logwarn("Failed to decode compressed image. Skipping save.")
            return

        try:
            os.makedirs(self._save_folder, exist_ok=True)
            timestamp = rospy.Time.now().to_sec()
            image_path = os.path.join(self._save_folder, f"rgb_{int(timestamp)}.png")
            cv2.imwrite(image_path, cv_image)
            rospy.loginfo(f"Saved RGB image to {image_path}")
        except OSError as e:
            rospy.logerr(f"Failed to write image to disk: {e}")


if __name__ == "__main__":
    try:
        republisher = ImageRepublisher()
        republisher.run()
    except rospy.ROSInterruptException:
        pass
