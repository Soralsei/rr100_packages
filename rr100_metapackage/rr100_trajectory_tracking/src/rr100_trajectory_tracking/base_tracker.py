
from abc import abstractmethod
from geometry_msgs.msg import PointStamped
import tf2_ros
import numpy as np


class BaseTrajectoryTracker:
    def __init__(self, goal: PointStamped, file_descriptor, buffer: tf2_ros.Buffer) -> None:
        self.goal = goal
        self.previous_position = None
        
        self.tf = buffer
        
        self.save_file = file_descriptor
        self.total_distance: float = 0.0

    @abstractmethod
    def _extract_position_from_feedback(self, feedback) -> np.ndarray:
        pass
 
    @abstractmethod
    def _is_result_success(self, result) -> bool:
        pass

    def feedback_cb(self, feedback):
        if self.previous_position is None:
            self.previous_position = self._extract_position_from_feedback(feedback)
            return

        current_position = self._extract_position_from_feedback(feedback)

        self.total_distance += np.linalg.norm(current_position - self.previous_position)
        self.previous_position = current_position

    def done_cb(self, _, result):
        if self._is_result_success(result):
            self.save_file.write(
                (f"[{self.__class__}] Goal (frame: {self.goal.header.frame_id}) = "
                 f"({self.goal.point.x, self.goal.point.y, self.goal.point.z})"
                 f" | total distance: {self.total_distance}\n")
            )
            print(f"Total distance traveled for goal {self.total_distance}")
            self.save_file.flush()
        self.previous_position = None
        self.total_distance = 0.0
