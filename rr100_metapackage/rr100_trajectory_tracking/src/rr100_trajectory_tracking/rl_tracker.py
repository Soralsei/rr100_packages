from typing import Optional
import numpy as np

from numpy.core import ndarray
from geometry_msgs.msg import PoseStamped
from rr100_rl_msgs.msg import GoToPoseResult, GoToPoseFeedback
from rr100_trajectory_tracking.base_tracker import BaseTrajectoryTracker
# from tf.transformations import euler_from_quaternion

class RLTracker(BaseTrajectoryTracker):

    def _extract_position_from_feedback(self, feedback) -> ndarray:
        return np.array(
            [feedback.current_pose.pose.position.x, feedback.current_pose.pose.position.y, feedback.current_pose.pose.position.z]
        )

    def _extract_pose_from_result(self, result: GoToPoseResult) -> Optional[PoseStamped]:
        return result.final_pose

    def _is_result_success(self, result) -> np.bool:
        return result.status == GoToPoseResult.SUCCEEDED