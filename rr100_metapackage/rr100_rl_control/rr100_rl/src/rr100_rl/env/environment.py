import time
from typing import Any, Iterable, Optional
import rospy
import numpy as np

from geometry_msgs.msg import TransformStamped, PointStamped, PoseStamped, TwistStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState

from tf2_ros import Buffer, TransformListener
import tf2_ros, tf2_geometry_msgs, tf2_kdl, PyKDL

# def euler_from_quaternion(quaternion: Quaternion) -> np.ndarray:
#     """
#     Convert a quaternion to Euler angles (roll, pitch, yaw).
#     """
#     quat = np.array([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
#     euler =np.zeros(3)
#     quat_squared = np.power(quat, 2)
#     unit = np.sum(quat_squared) # 1 if normalized, correction if not
#     test = quat[0] * quat[1] + quat[2] * quat[3]
    
#     if test > 0.499 * unit:
#         euler[0] = 0
#         euler[1] = np.pi / 2
#         euler[2] = 2 * atan2(quat[0], quat[3])
#         return euler
    
#     if test < -0.499 * unit:
#         euler[0] = 0
#         euler[1] = -np.pi / 2
#         euler[2] = -2 * atan2(quat[0], quat[3])
#         return euler
        
#     euler[0] = atan2(2*quat[0]*quat[3]-2*quat[1]*quat[2] , -quat_squared[0] + quat_squared[1] - quat_squared[2] + quat_squared[3])
#     euler[1] = math.asin(2*test/unit);
#     euler[2] = atan2(2*quat[1]*quat[3]-2*quat[0]*quat[2] , quat_squared[0] - quat_squared[1] - quat_squared[2] + quat_squared[3]);
    
#     return euler

class Environment:
    
    def __init__(
        self, 
        num_observation: int, 
        global_frame: str, 
        base_frame: str,
        wheel_joint_names: str,
        steering_joint_names: str,
        odom_topic: str,
        cmd_vel_topic: str,
        joint_states_topic: str,
        max_velocity: Iterable[Any],
        max_acceleration: Iterable[Any],
        max_xy_distance: float = 2.0,
        action_frequency: int = 40,
    ) -> None:
        self.max_xy_distance = max_xy_distance
        self.num_observation = num_observation
        self.global_frame = global_frame
        self.base_frame = base_frame

        rospy.loginfo(f"[Environment] Global frame : {self.global_frame}")
        rospy.loginfo(f"[Environment] Base frame : {self.base_frame}")

        self.wheel_joint_names = wheel_joint_names
        self.steering_joint_names = steering_joint_names

        self.max_velocity = np.array(max_velocity)          # [max_L_x, max_W_z]
        self.max_acceleration = np.array(max_acceleration)  # [max_L_x, max_W_z]

        rospy.loginfo(f"[Environment] Max velocity : {max_velocity}")
        rospy.loginfo(f"[Environment] Max acceleration : {max_acceleration}")
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer) # Position, orientation
        
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_cb)
        self.joint_states_sub = rospy.Subscriber(joint_states_topic, JointState, self.joint_states_cb)
        self.cmd_vel_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=100)

        self.last_odom: Optional[Odometry] = None           # Velocity, angular velocity
        self.last_joint_states: Optional[JointState] = None # Wheel radial velocity, Steering angle, Steering velocity
        
        self.goal: Optional[PointStamped] = None
        self.initial_robot_tf: Optional[TransformStamped] = None
        self.current_robot_tf: Optional[TransformStamped] = None

        self.action_rate = rospy.Rate(action_frequency)
        self.dt = self.action_rate.sleep_dur.to_sec()
      
      
    def initialize(self, max_retries : int,):
      for i in range(max_retries):
        self.current_robot_tf = self.lookup_robot_tf()
        if self.current_robot_tf is not None:
          return True
        rospy.logerr(f"Environment initialization : missing robot TF, retrying...")
        time.sleep(2.0)
      return False
      
    
    def odom_cb(self, msg: Odometry) -> None:
        self.last_odom = msg
    
    
    def joint_states_cb(self, msg: JointState) -> None:
        self.last_joint_states = msg
    
    
    def get_observation(self) -> Optional[np.ndarray]:
        robot_tf = self.lookup_robot_tf()
        if robot_tf is None or self.last_odom is None is None or self.last_joint_states is None:
            rospy.logerr("Robot transform or odometry data not yet available")
            return None
        
        if self.goal is None:
            return None
        
        self.current_robot_tf = robot_tf
        
        wheel_radial, steering_angle, steering_rate = self.get_wheel_info()
        # linear_velocity, angular_velocity = self.get_robot_velocity()
        robot_pose = np.zeros(2)
        twist = PyKDL.Twist()
        robot_yaw = 0
        if self.initial_robot_tf is not None:
            robot_tf = tf2_kdl.transform_to_kdl(robot_tf)
            twist = PyKDL.Twist(
                PyKDL.Vector(self.last_odom.twist.twist.linear.x, self.last_odom.twist.twist.linear.y, self.last_odom.twist.twist.linear.z),
                PyKDL.Vector(self.last_odom.twist.twist.angular.x, self.last_odom.twist.twist.angular.y, self.last_odom.twist.twist.angular.z)
            )
            # print(twist)
            initial = tf2_kdl.transform_to_kdl(self.initial_robot_tf)
            # rospy.loginfo(f"Current frame : \n{robot_tf}")
            # rospy.loginfo(f"Initial frame : \n{initial}")
            # rospy.loginfo(f"Initial frame : \n{initial.Inverse()}")
            initial_inv = initial.Inverse()

            robot_in_initial = initial_inv * robot_tf
            robot_pose[0] = robot_in_initial.p.x()
            robot_pose[1] = robot_in_initial.p.y()

            twist = robot_tf.M * twist
            # print(f"Transformed twist : {twist}")
            
            robot_yaw = robot_in_initial.M.GetRPY()[2]

        self.relative_robot_tf = robot_pose.copy()

        # rospy.loginfo(f"Current robot pose (relative to initial pose) : {robot_pose}")
        # rospy.logdebug(f"Current robot orientation (relative to initial pose) : {robot_yaw}")
        
        goal = np.array([
            self.goal.point.x,
            self.goal.point.y,
        ])
        
        # Robot pose, goal, wheel velocity, steering angle, steering rate, linear velocity, yaw, angular velocity
        return np.concatenate(
            (
                robot_pose,
                goal,
                wheel_radial,
                steering_angle,
                steering_rate,
                # [4e-5],
                [twist.vel.x(), twist.vel.y()],
                [robot_yaw],
                [twist.rot.z()]
            )
        ) # type: ignore
    
    
    def get_goal(self, goal: PointStamped) -> Optional[PointStamped]:
        # Get relative goal position
        try:
            goal = self.tf_buffer.transform(goal, self.base_frame)
        except Exception as e:
            rospy.logerr(f"Failed to transform goal to frame '{self.base_frame}': {e}")
            return None
        # goal_to_robot = self.lookup_transform(goal.header.frame_id, self.base_frame)
        # if goal_to_robot is None:
        #     rospy.logerr("Could not find transform from goal to robot")
        #     return None        
        
        self.goal = goal
        
        if abs(self.goal.point.x) > self.max_xy_distance or abs(self.goal.point.y) > self.max_xy_distance:
            rospy.logwarn(f">>> Goal ({goal.point.x:.3f}, {goal.point.y:.3f}) is too far away (limit is {self.max_xy_distance, self.max_xy_distance})")
            return None
        
        return self.goal
    
    
    def lookup_robot_tf(self) -> Optional[TransformStamped]:
        self.current_robot_tf = self.lookup_transform(self.base_frame, self.global_frame)
        return self.current_robot_tf
        
        
    def lookup_transform(self, source: str, dest: str):
        # rospy.loginfo(f"{source} -> {dest}")
        try:
            return self.tf_buffer.lookup_transform(dest, source, rospy.Time(0), rospy.Duration(1))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e: # type: ignore
            rospy.logerr(f"Error looking up transform: {e}")
            return None
    
    
    def step(self, action: np.ndarray):
        self.set_action(action)
        self.action_rate.sleep()
        obs = self.get_observation()
        return obs
    
    
    def set_action(self, action):
        action_scaled = action * self.max_velocity
        action_scaled = np.clip(action_scaled, -self.max_velocity, self.max_velocity)
        action_scaled = self.limit_action(action_scaled, self.previous_action, max_acceleration=self.max_acceleration, dt=self.dt)
        rospy.logdebug(f"action_scaled = {action_scaled}")
        cmd = Twist()
        cmd.linear.x = action_scaled[0]
        cmd.angular.z = action_scaled[1]
        self.cmd_vel_pub.publish(cmd)

        self.previous_action = action_scaled

    
    def limit_action(self, action, prev_action, max_acceleration, dt) -> np.ndarray:
        """
        Limite la variation entre current_cmd et previous_cmd.
        :param current_cmd: Commande souhaitée actuelle (par exemple, vitesse ou angle).
        :param previous_cmd: Commande appliquée lors du pas précédent.
        :param max_acceleration: Variation maximale autorisée par seconde.
        :param dt: Intervalle de temps entre deux pas de simulation.
        :return: Commande lissée à appliquer.
        """
        # Calcul de la variation souhaitée
        delta = action - prev_action
        # Variation maximale autorisée durant dt
        max_delta = max_acceleration * dt
        # Limitation de la variation
        delta_limited = np.clip(delta, -max_delta, max_delta)
        # Commande finale
        return prev_action + delta_limited
    
    
    def reset(self):
        self.initial_robot_tf = self.lookup_robot_tf()
        self.previous_action = np.zeros(2)
        rospy.logdebug(f"Initial robot tf : {self.initial_robot_tf}")
        return self.get_observation()
        
        
    def get_wheel_info(self) -> tuple:
        '''
        Returns wheel state tuple in the following order : (wheel velocity, steering position, steering velocity)
        '''
        vel_wheel, pos_steer, vel_steer = np.zeros((2,)), np.zeros((2,)), np.zeros((2,))
        if self.last_joint_states is None:
            return vel_wheel, pos_steer, vel_steer
        
        # rospy.loginfo(self.last_joint_states)
        i, j = 0, 0

        for name, position, velocity in zip(
            self.last_joint_states.name, 
            self.last_joint_states.position, 
            self.last_joint_states.velocity
        ):
            if name in self.wheel_joint_names:
                vel_wheel[i] = velocity
                i += 1
            elif name in self.steering_joint_names:
                pos_steer[j] = position
                vel_steer[j] = velocity
                j += 1
                
        return vel_wheel, pos_steer, vel_steer
    
    
    def get_robot_velocity(self) -> tuple:
        linear, angular = np.zeros((1,)), np.zeros((1,))
        if self.last_odom is None:
            return linear, angular
        
        # rospy.loginfo(self.last_odom.twist)
        
        linear = np.array([self.last_odom.twist.twist.linear.x, self.last_odom.twist.twist.linear.y])
        angular = np.array([self.last_odom.twist.twist.angular.z])
        
        return linear, angular
    
    
    @property
    def distance_to_goal(self) -> float:
        if self.goal is None or self.relative_robot_tf is None:
            return float("nan")
        

        x = self.goal.point.x - self.relative_robot_tf[0]
        y = self.goal.point.y - self.relative_robot_tf[1]
        
        return np.linalg.norm(np.array([x, y])) # type: ignore
    
    
    @property
    def robot_pose(self) -> PoseStamped:
        if self.goal is None or self.current_robot_tf is None:
            return PoseStamped()
        pose = PoseStamped()
        pose.header = self.current_robot_tf.header
        pose.pose.position.x = self.current_robot_tf.transform.translation.x
        pose.pose.position.y = self.current_robot_tf.transform.translation.y
        pose.pose.position.z = 0.0
        
        return pose
    
    def stop(self):
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)