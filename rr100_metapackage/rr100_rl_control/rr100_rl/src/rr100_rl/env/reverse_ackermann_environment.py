from typing import Any, Iterable

import math
import rospy
import numpy as np

from geometry_msgs.msg import Twist
from rr100_rl.env.environment import Environment

class ReverseAckermannEnvironment(Environment):
  def __init__(self, num_observation: np.int, global_frame: np.str, base_frame: np.str, wheel_joint_names: np.str, steering_joint_names: np.str, odom_topic: np.str, cmd_vel_topic: np.str, joint_states_topic: np.str, max_velocity: Iterable[Any], max_acceleration: Iterable[Any], max_xy_distance: np.float = 2, action_frequency: np.int = 40) -> None:
    super().__init__(num_observation, global_frame, base_frame, wheel_joint_names, steering_joint_names, odom_topic, cmd_vel_topic, joint_states_topic, max_velocity, max_acceleration, max_xy_distance, action_frequency)
    self.wheel_radius = rospy.get_param("/rr100_steering_controller/wheel_radius", 0.21)
    self.wheel_base = rospy.get_param("/rr100_steering_controller/wheel_base") / 2
    rospy.loginfo("RL Ackermann environment")
    

  def set_action(self, action):
    # wheel_perimeter = 2 * np.pi * self.wheel_radius
    scaled_action = np.clip(
      action * self.max_velocity,
      -self.max_velocity,
      self.max_velocity
    )
    rospy.logdebug(f"Clipped action : {scaled_action}")
    
    scaled_action = self.limit_action(scaled_action, self.previous_action, max_acceleration=self.max_acceleration, dt=self.dt)
    rospy.logdebug(f"action_scaled = {scaled_action}")
    
    self.previous_action = scaled_action
    
    v = self.wheel_radius * scaled_action[0] * math.cos(scaled_action[1])
    w = self.wheel_radius * scaled_action[0] * math.sin(scaled_action[1]) / self.wheel_base
    
    applied_action = np.array([v, w])
    
    rospy.logdebug(f"Action after conversion to v and w : {applied_action}")
    
    cmd = Twist()
    cmd.linear.x = applied_action[0]
    cmd.angular.z = applied_action[1]
    self.cmd_vel_pub.publish(cmd)

  