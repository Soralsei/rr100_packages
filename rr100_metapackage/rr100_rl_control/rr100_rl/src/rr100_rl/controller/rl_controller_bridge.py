import time
import rospy
import zmq
import numpy as np

from actionlib.simple_action_server import SimpleActionServer
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker

from rr100_rl.env.environment import Environment
from rr100_rl_msgs.msg import GoToPoseAction
from rr100_rl_msgs.msg import GoToPoseGoal, GoToPoseResult, GoToPoseFeedback
from geometry_msgs.msg import PoseStamped, PointStamped


class RLControllerBridge:
    '''
    RL controller bridge between ROS and JAX-based RL frameworks
    Contains a ZMQ client that sends requests containing the state 
    to the RLController and expects actions in responses.
    Cf. rl_controller.py for reason for this architecture
    '''
    def __init__(
        self,
        env: Environment,
        goal_marker_topic: str = "rl_controller/goal_marker",
        action_server_name: str = "rl_controller",
        distance_threshold: float = 0.10,
        goal_max_iter: int = 500,
    ) -> None:
        self.env: Environment = env
        self.action_name = action_server_name

        self.distance_threshold = distance_threshold
        self.goal_max_iter = goal_max_iter

        self.action_server = SimpleActionServer(
            action_server_name, GoToPoseAction, self.goto_cb, auto_start=False
        )
        self.goal_marker_pub = rospy.Publisher(goal_marker_topic, Marker, queue_size=1)

        env.initialize(10)

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        # self.socket.setsockopt(zmq.LINGER, 0)
        # self.socket.setsockopt(zmq.IMMEDIATE, 1)
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)

        # self.server_uri = f"tcp://localhost:{rr100_rl.CONTROLLER_PORT}"
        self.server_uri = f"ipc:///tmp/rl_controller.pipe"
        rospy.loginfo(f"Connecting to ZMQ RL controller at '{self.server_uri}'...")
        self.socket.connect(self.server_uri)

        rospy.loginfo("Starting ActionServer")
        self.action_server.start()

        self.debug = False
        if self.debug:
            self.gazebo_pause_srv = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
            self.gazebo_unpause_srv = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
            rospy.loginfo("Waiting for gazebo play/unpause services...")
            self.gazebo_pause_srv.wait_for_service()
            self.gazebo_unpause_srv.wait_for_service()
            rospy.loginfo("Done")

    def end(self):
        self.socket.disconnect(self.server_uri)
        self.socket.close()
        self.context.term()

    def goto_cb(self, goal: GoToPoseGoal) -> None:
        g = self.env.get_goal(goal.goal)
        obs = self.env.reset()
        if g is None or obs is None:
            result = GoToPoseResult(PoseStamped(), GoToPoseResult.OOREACH)
            self.action_server.set_aborted(
                result=result, text=f"Fail to get goal or observation"
            )
            return
        rospy.loginfo(f"Got goal {goal.goal.point.x, goal.goal.point.y} in frame '{goal.goal.header.frame_id}'")
        self.publish_goal_marker(goal.goal)

        rospy.logdebug(f"First observation : {obs}")

        result = GoToPoseResult()
        result.status = GoToPoseResult.FAILED

        for i in range(self.goal_max_iter):
            if self.debug:
                self.gazebo_unpause_srv.call()

            feedback = GoToPoseFeedback()
            feedback.current_pose = self.env.robot_pose
            self.action_server.publish_feedback(feedback=feedback)
            if self.action_server.is_preempt_requested():
                rospy.loginfo(f"{self.action_name} : Prempted")
                self.action_server.set_preempted()
                self.env.stop()
                return
            
            d = self.env.distance_to_goal
            rospy.logdebug(f"Current distance to goal : {d}")
            if d < self.distance_threshold:
                rospy.loginfo(f"Target reached ! ({d})")
                result.final_pose = self.env.robot_pose
                result.status = GoToPoseResult.SUCCEEDED
                self.action_server.set_succeeded(result=result)
                self.env.stop()
                return

            req = {"observation": obs.tolist(), "episode_start": None, "deterministic": False} # type: ignore
            # rospy.loginfo(f"Feed back : {feedback}")
            rospy.logdebug(f"Sending request {req}")
            t1 = time.monotonic()
            self.socket.send_json(req)
            # rospy.logdebug(f"Waiting for response...")
            try:
                response = self.socket.recv_json()
            except Exception as e:
                rospy.logerr(f"Failed to receive action from controller : {e}")
                self.action_server.set_aborted()
                return
            elapsed = time.monotonic() - t1
            rospy.logdebug(f"REQ -> REP RTT : {elapsed * 1000.0 :.2f}ms")
            rospy.logdebug(f'Got action {response["action"]}') # type: ignore
            obs = self.env.step(np.array(response["action"])) # type: ignore
            
            if self.debug:
                self.gazebo_pause_srv.call()
                input("Press enter to continue...")

        self.env.stop()
        dist = self.env.distance_to_goal
        if dist < self.distance_threshold:
            rospy.loginfo(f"Target reached ! ({dist}m)")
            result.final_pose = self.env.robot_pose
            result.status = GoToPoseResult.SUCCEEDED
        else:
            rospy.loginfo(f"Target not reached ! ({d}m)")

        self.action_server.set_succeeded(result)

    def get_goal_marker(self, goal: PointStamped) -> Marker:
        marker = Marker()
        marker.action = marker.ADD

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        marker.type = Marker.SPHERE
        marker.id = 0
        marker.header.frame_id = goal.header.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "rl_controller"
        marker.pose.position = goal.point
        marker.pose.position.z = 1.2
        marker.pose.orientation.w = 1.0

        return marker
    
    def publish_goal_marker(self, goal: PointStamped):
        marker = self.get_goal_marker(goal)
        self.goal_marker_pub.publish(marker)