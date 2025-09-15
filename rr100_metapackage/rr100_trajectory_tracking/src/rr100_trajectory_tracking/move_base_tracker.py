import numpy as np
from geometry_msgs.msg import PointStamped, PoseStamped
from move_base_msgs.msg import MoveBaseResult, MoveBaseFeedback
from rr100_trajectory_tracking.base_tracker import BaseTrajectoryTracker
from tf2_ros.buffer import Buffer
from tf.transformations import euler_from_quaternion


class MoveBaseTracker(BaseTrajectoryTracker):

    def __init__(self, goal: PointStamped, file_descriptor, buffer: Buffer, comment: str="") -> None:
        super().__init__(goal, file_descriptor, buffer, comment)
        self.previous_pose: PoseStamped = None

    def _extract_position_from_feedback(self, feedback) -> np.ndarray:
        self.previous_pose = feedback.base_position
        return np.array([
            self.previous_pose.pose.position.x,
            self.previous_pose.pose.position.y,
            self.previous_pose.pose.position.z
        ])
        
    def _extract_pose_from_result(self, _: MoveBaseResult) -> np.ndarray:
        print(f"Final pose: {self.previous_pose}")
        return self.previous_pose # type: ignore

    def _is_result_success(self, _: MoveBaseResult) -> np.bool:
        return True