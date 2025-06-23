#!/usr/bin/env python3

import rospy
from dynamic_reconfigure import server
from rr100_drive_amp.cfg import DriveAmpConfig

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Vector3

import numpy as np

import conversion
from threading import Lock

import sys

class DriveAmp():
    """
    Class that tries to correct a velocity command using a PID controller
    to better match a target velocity by listening to odometry information.
    
    * Subscribes to ~/twist_in and ~/odom_in
    * Publishes to ~/corrected_cmd_vel
    
    ROS params : 
    * ~/gains : dictionary of 3 key/value pairs (p, i and d)
    * ~/frequency : float used to control publishing frequency
    """
    def __init__(self) -> None:
        """
        Constructor.
        Takes no parameters, returns a DriveAmp
        """
        self.active = True
        self.reconf_server = server.Server(DriveAmpConfig, self.reconfigure)
        
        gains = rospy.get_param('gains', {'p' : 1.0, 'i' : 0.0, 'd' : 0.0})
        
        self.target = np.zeros((2, 3), dtype=np.float32)
        self.current = np.zeros((2, 3), dtype=np.float32)
        
        self.k_p, self.k_i, self.k_d = gains['p'], gains['i'], gains['d']
        self.cumulated_error = np.zeros((2, 3), dtype=np.float32)
        self.previous_error = np.zeros((2, 3), dtype=np.float32)
        
        # self.timer = rospy.Timer(rospy.Duration(1.0 / frequency), self.update)
        
        self.pub = rospy.Publisher("corrected_cmd_vel", Twist, queue_size=20)
        self.twist_sub = rospy.Subscriber("twist_in", Twist, self._twist_callback, queue_size=10)
        self.odom_sub = rospy.Subscriber("odom_in", Odometry, self._odom_callback, queue_size=10)
        
        self.receive_lock = Lock()
        self.integrator_lock = Lock()
        
        self.previous_time = rospy.get_rostime()
    
    def reconfigure(self, config, _) -> None:            
        self.active = config.active
        
        self.k_p = config.p_gain
        self.k_i = config.i_gain
        self.k_d = config.d_gain
        
        return config
    
    def _twist_callback(self, twist: Twist) -> None:
        is_close = False
        with self.receive_lock:
            self.target = np.reshape(np.concatenate(conversion.kinematics_to_numpy(twist), axis=0), (2, 3))
            is_close = np.allclose(self.target, self.current, atol=1e-3)
           
        if not is_close: 
            with self.integrator_lock:
                self.cumulated_error.fill(0)
    
    def _odom_callback(self, odometry: Odometry) -> None:
        current_time = rospy.get_rostime()
        
        target = None
        with self.receive_lock:
            target = self.target.copy()
            
        if not self.active:
            # If the node is not active, just forward 
            # the twist_in command velocities
            self.pub.publish(target)
            return
        
        linear, angular, _ = conversion.kinematics_with_covariance_to_numpy(odometry.twist)
        with self.receive_lock:
            if np.allclose(target, 0, atol=1e-5): 
                self.target = np.zeros((2, 3), dtype=np.float32)
                message = Twist()
                self.pub.publish(message)
                return
            self.current = np.reshape(np.concatenate((linear, angular)), (2, 3))
        
            
        dt = (current_time - self.previous_time).to_sec()
        
        if abs(dt <= 1e-9):
            return
        
        with self.integrator_lock:
            error = target - self.current
            self.cumulated_error += error * dt
            error_rate = (error - self.previous_error) / dt
        
            correction = self.k_p * error + self.k_i * self.cumulated_error + self.k_d * error_rate
            
        corrected_linear, corrected_angular = target + correction
        corrected_linear[1:] = 0.
        corrected_angular[:-1] = 0.
        
        rospy.logdebug(f"Ep = {error}")
        rospy.logdebug(f"Ei = {self.cumulated_error}")
        rospy.logdebug(f"Ed = {error_rate}")
        rospy.logdebug(f"Target linear :{target[0]}\nTarget angular :{target[1]}")
        rospy.logdebug(f"Corrected linear :{corrected_linear}\nCorrected angular :{corrected_angular}")
        
        message = conversion.numpy_to_kinematics(Twist, corrected_linear, corrected_angular)
        # rospy.loginfo(f"[DriveAmp] Corrected velocity command : {message}")
    
        self.pub.publish(message)
    
        self.previous_error = error
        self.previous_time = current_time
        
        
if __name__ == '__main__':
    rospy.init_node('drive_amp', sys.argv, log_level=rospy.INFO)
    amp = DriveAmp()
    rospy.spin()