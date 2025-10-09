
from abc import abstractmethod
from typing import Optional
from csv import DictWriter

from tf.transformations import euler_from_quaternion

from geometry_msgs.msg import PointStamped, PoseStamped

import rospy
import tf2_ros
import tf2_geometry_msgs
import numpy as np



class BaseTrajectoryTracker:
    def __init__(
        self,
        goal: PointStamped,
        file_descriptor,
        buffer: tf2_ros.Buffer,
        comment: Optional[str] = None
    ) -> None:
        self.tf = buffer
        self.goal = self.tf.transform(
            goal,
            target_frame="odom",
            timeout=rospy.Duration(secs=1),
        )
        self.goal_in_base = self.tf.transform(
            goal,
            target_frame="base_footprint",
            timeout=rospy.Duration(secs=1),
        )
        self.initial_distance = np.linalg.norm(
            np.array([self.goal_in_base.point.x, self.goal_in_base.point.y])
        )
        self.previous_position = None
        self.comment = comment
        
        self.save_file = file_descriptor
        self.total_traveled: float = 0.0
        

    @abstractmethod
    def _extract_position_from_feedback(self, feedback) -> np.ndarray:
        pass

    @abstractmethod
    def _extract_pose_from_result(self, _) -> PoseStamped:
        pass
 
    @abstractmethod
    def _is_result_success(self, result) -> bool:
        pass

    def _feedback_cb(self, feedback):
        if self.previous_position is None:
            self.previous_position = self._extract_position_from_feedback(feedback)
            return

        current_position = self._extract_position_from_feedback(feedback)

        self.total_traveled += np.linalg.norm(current_position - self.previous_position)
        self.previous_position = current_position

    def _done_cb(self, _, result):
        writer = DictWriter(
            self.save_file,
            fieldnames=[
                "class", 
                "goal_frame", "goal_x", "goal_y",
                "initial_distance", "total_traveled", "success",
                "final_frame", "final_x", "final_y", "final_theta",
                "distance_to_goal", "comment",
            ]
        )
        if self.save_file.tell() == 0:
            writer.writeheader()

        final_pose = self._extract_pose_from_result(result)
        yaw = euler_from_quaternion([
                final_pose.pose.orientation.x,
                final_pose.pose.orientation.y,
                final_pose.pose.orientation.z,
                final_pose.pose.orientation.w,
        ])[2]
        
        got_transform = False
        goal: np.ndarray = np.array([np.nan, np.nan])
        final_position: np.ndarray = np.array([np.nan, np.nan])
        
        while not rospy.is_shutdown() and not got_transform:
            try: 
                goal_in_final = self.tf.transform(
                    self.goal,
                    target_frame=final_pose.header.frame_id,
                    timeout=rospy.Duration(secs=1),
                )
                goal: np.ndarray = np.array([goal_in_final.point.x, goal_in_final.point.y])
                final_position: np.ndarray = np.array([final_pose.pose.position.x, final_pose.pose.position.y])
                got_transform = True
            
            except Exception as e:
                rospy.logwarn(f"Could not transform goal from frame '{self.goal.header.frame_id}' to frame '{final_pose.header.frame_id}': {e}")
        final_distance = np.linalg.norm(final_position - goal)
        writer.writerow(
            {
                "class": self.__class__.__name__,
                "goal_frame": self.goal.header.frame_id,
                "goal_x": self.goal.point.x,
                "goal_y": self.goal.point.y,
                "initial_distance": self.initial_distance,
                "total_traveled": self.total_traveled,
                "success": self._is_result_success(result),
                "final_frame": final_pose.header.frame_id,
                "final_x": final_pose.pose.position.x,
                "final_y": final_pose.pose.position.y,
                "final_theta": yaw,
                "distance_to_goal": final_distance,
                "comment": self.comment
            }
        )
        print(f"Total distance traveled for goal {self.total_traveled}")
        self.save_file.flush()
        self.previous_position = None
        self.total_traveled = 0.0
