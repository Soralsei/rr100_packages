import numpy as np

from numpy.core import ndarray
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseResult, MoveBaseFeedback
from rr100_trajectory_tracking.base_tracker import BaseTrajectoryTracker


class MoveBaseTracker(BaseTrajectoryTracker):

    def _extract_position_from_feedback(self, feedback) -> ndarray:
        return np.array(
            [feedback.base_position.pose.position.x, feedback.base_position.pose.position.y, feedback.base_position.pose.position.z]
        )

    def _is_result_success(self, result) -> np.bool:
        return True