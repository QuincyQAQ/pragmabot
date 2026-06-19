"""Scene observation via synchronized ROS camera topics."""

import logging
import time
from typing import Tuple

import cv2
import numpy as np
from omegaconf import DictConfig
from PIL import Image

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo as CameraInfoMsg
from sensor_msgs.msg import CompressedImage as CompressedImageMsg
from sensor_msgs.msg import Image as ImageMsg
import message_filters

from pragmabot.geometry import CameraIntrinsics

logger = logging.getLogger(__name__)


class SceneObserver:
    """Capture synchronized color/depth images and camera intrinsics from ROS topics."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize subscribers for synchronized camera topics.

        Args:
            config: Configuration object containing ``color_image``,
                ``depth_image``, and ``camera_info``.
        """
        self.color_image_topic = config.color_image
        self.depth_image_topic = config.depth_image
        self.camera_info_topic = config.camera_info

        self.latest_obs = None
        self.request_pending = False

        logger.info("Establishing continuous synchronized connections...")
        self.color_sub = message_filters.Subscriber(self.color_image_topic, CompressedImageMsg)
        self.depth_sub = message_filters.Subscriber(self.depth_image_topic, ImageMsg)
        self.info_sub = message_filters.Subscriber(self.camera_info_topic, CameraInfoMsg)

        self.time_sync = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.info_sub], queue_size=10, slop=1.0
        )
        self.time_sync.registerCallback(self._sync_callback)

        self.bridge = CvBridge()

    def get_scene_observation(self) -> Tuple[Image.Image, Image.Image, CameraIntrinsics, rospy.Time]:
        """Fetch a fresh synchronized color, depth, and info observation."""

        # Clear old data and open the gate for the callback
        self.latest_obs = None
        self.request_pending = True

        # Poll until the sync callback delivers a fresh frame
        while self.latest_obs is None:
            time.sleep(0.02)

        color_msg, depth_msg, info_msg = self.latest_obs
        observation_time = depth_msg.header.stamp  # Use depth timestamp as the observation time

        # Build intrinsics from the camera info message
        intrinsics = CameraIntrinsics(
            width=info_msg.width,
            height=info_msg.height,
            fx=info_msg.K[0],
            fy=info_msg.K[4],
            ppx=info_msg.K[2],
            ppy=info_msg.K[5],
        )

        # Process color image
        color_bytes = np.frombuffer(color_msg.data, np.uint8)
        color_bgr = cv2.imdecode(color_bytes, cv2.IMREAD_COLOR)
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        color_image = Image.fromarray(color_rgb)

        # Process depth image
        depth_array = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth_image = Image.fromarray(depth_array)
        return color_image, depth_image, intrinsics, observation_time

    def _sync_callback(self, color_msg: CompressedImageMsg, depth_msg: ImageMsg, info_msg: CameraInfoMsg) -> None:
        """Only update the synchronized triplet if actively requested."""
        if self.request_pending:
            self.latest_obs = (color_msg, depth_msg, info_msg)
            self.request_pending = False  # Instantly close the gate so we only grab one frame
