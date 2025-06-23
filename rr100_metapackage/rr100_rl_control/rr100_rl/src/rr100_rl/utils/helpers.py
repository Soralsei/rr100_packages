import copy
import glob
import os
import string
import sys

from typing import Any, Iterable, Union, Mapping, AnyStr

import numpy as np
from torch import AnyType

import rospy

#################################################
# Type conversion functions #####################
#################################################
def model_state_msg_2_link_state_dict(link_state_msgs):
    """Converts the a `gazebo_msgs/ModelState <https://docs.ros.org/en/jade/api/gazebo_msgs/html/msg/ModelState.html>`_
    message into a state dictionary. Contrary to the original ModelState message,
    in the model state dictionary the poses and twists are grouped per link/model.

    Args:
        link_state_msgs (:obj:`gazebo_msgs.msg.ModelState`): A ModelState message.

    Returns:
        dict: A model_state dictionary.
    """  # noqa: E501
    model_state_dict = {}
    for joint_name, position, twist in zip(
        link_state_msgs.name, link_state_msgs.pose, link_state_msgs.twist
    ):
        model_state_dict[joint_name] = {}
        model_state_dict[joint_name]["pose"] = copy.deepcopy(position)
        model_state_dict[joint_name]["twist"] = copy.deepcopy(twist)

    return model_state_dict


def pose_msg_2_pose_dict(pose_msg):
    """Create a pose dictionary ``{x, y, z, rx, ry, rz, rw}`` out of a
    `geometry_msgs.msg.Pose <https://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/Pose.html>`_
    message.

    Args:
        pose_msg (:obj:`geometry_msgs.msg.Pose`): A pose message

    Returns:
        dict: Dictionary that contains the pose.
    """  # noqa: E501
    pose_dict = {
        "x": pose_msg.position.x,
        "y": pose_msg.position.y,
        "z": pose_msg.position.z,
        "rx": pose_msg.orientation.x,
        "ry": pose_msg.orientation.y,
        "rz": pose_msg.orientation.z,
        "rw": pose_msg.orientation.w,
    }

    return pose_dict


def flatten(iterator: Union[Any, Iterable]):
    if isinstance(iterator, Iterable):
        for it in iterator:
            yield from flatten(it)
    else:
        yield iterator
        
def ros_exit_gracefully(shutdown_msg=None, exit_code=0):
    """Shuts down the ROS node wait until it is shutdown and exits the script.

    Args:
        shutdown_msg (str, optional): The shutdown message. Defaults to ``None``.
        exit_code (int, optional): The exit code. Defaults to ``0``.
    """
    if exit_code == 0:
        rospy.loginfo(shutdown_msg)
    else:
        rospy.logerr(shutdown_msg)
    rospy.signal_shutdown(shutdown_msg)
    while not rospy.is_shutdown():
        rospy.sleep(0.1)
    sys.exit(exit_code)
    
def deep_update(d, u=None, fixed=False, **kwargs):  # noqa: C901
    """Updates a dictionary recursively (i.e. deep update). This function takes a update
    dictionary and/or keyword arguments. When a keyword argument is supplied, the
    key-value pair is changed if it exists somewhere in the dictionary.

    Args:
        d (dict): Dictionary you want to update.
        u (dict, optional): The update dictionary.
        fixed (bool, optional): Whether you want the input dictionary to be fixed
            (i.e. only change keys that are already present). Defaults to ``False``.
        **kwargs: Keyword arguments used for creating the dictionary keys.

    Returns:
        dict: The updated dictionary.

    .. seealso::
        Based on the answer given by `@alex-martelli <https://stackoverflow.com/users/95810/alex-martelli>`_
        on `this stackoverflow question <https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth>`_.
    """  # noqa: E501
    # Update dict based on input dictionary.
    if u:
        for k, v in u.items():
            if isinstance(v, Mapping):
                if k in d.keys() or not fixed:
                    d[k] = deep_update(d.get(k, {}), v, fixed=fixed)
            else:
                if k in d.keys() or not fixed:
                    d[k] = v

    # Update dict based on keyword arguments.
    for key, val in kwargs.items():
        for k, v in d.items():
            if isinstance(v, Mapping):
                d[k] = deep_update(v, fixed=fixed, **{key: val})
            else:
                if k == key and key in d.keys():
                    d[k] = val

    # Print warning if no update dictionary or keyword argument was supplied.
    if not kwargs and not u:
        rospy.logwarn(
            "Returning original dictionary since no update dictionary or keyword "
            "argument was supplied."
        )

    return d

def find_gazebo_model_path(model_name, models_directory_path, extension=""):
    """Finds the path of the ``sdf`` or ``urdf`` file that belongs to a given
    ``model_name``. This is done by searching in the ``models_directory_path`` folder.
    If no file was found the model file path is returned empty.

    Args:
        model_name (str): The name of the model for which you want to find the path.
        models_directory_path (str): The path of the folder that contains the gazebo
            models. extension (str, optional): The model path extension. Defaults to
            ``""`` meaning that the function will first look for a ``sdf`` file and if
            that is not found it will look for a ``urdf`` file.

    Returns:
        (tuple): tuple containing:

            - :obj:`str`: The path where the ``sdf`` or ``urdf`` model file can be
              found.
            - :obj:`str`: Extension of the model file.
    """
    if extension and not extension.startswith("."):
        extension = "." + extension

    # Try to find the model path for a given model_name.
    model_directory_path = os.path.join(models_directory_path, model_name)
    if os.path.isdir(model_directory_path):  # Check if model directory exists.
        if extension != "":
            model_path = glob.glob(
                os.path.join(model_directory_path, "model" + extension)
            )
            if model_path:
                return model_path[0]
        else:  # no extension given.
            for ext in [".sdf", ".urdf"]:
                model_path = glob.glob(
                    os.path.join(model_directory_path, "model" + ext)
                )
                if model_path:
                    return model_path[0], ext[1:]

    # If model path could not be found.
    rospy.logwarn(
        f"Model path for '{model_name}' could not be found. Please check if the "
        f"'{model_name}.sdf' or '{model_name}.urdf' file exist in the model directory "
        f"'{model_directory_path}'."
    )
    return "", ""

def lower_first(string: str) -> str:
    return f"{string[0].lower()}{string[1:]}"

def quaternion_norm(quaternion) -> np.floating[Any]:
    return np.linalg.norm(quaternion)

def normalize_quaternion(quaternion):
    """Normalizes a given quaternion.

    Args:
        quaternion (:obj:`geometry_msgs.msg.Quaternion`): A quaternion.

    Returns:
        :obj:`geometry_msgs.msg.Quaternion`: The normalized quaternion.
    """
    quaternion = copy.deepcopy(
        quaternion
    )  # Make sure the original object is not changed.
    norm = quaternion_norm(quaternion)

    # Normalize quaternion.
    if np.isnan(norm):
        rospy.logwarn(
            "Quaternion could not be normalized since the norm could not be "
            "calculated."
        )
    elif norm == 0.0:  # Transform to identity.
        quaternion.x = 0.0
        quaternion.y = 0.0
        quaternion.z = 0.0
        quaternion.w = 1.0
    else:
        quaternion.x = quaternion.x / norm
        quaternion.y = quaternion.y / norm
        quaternion.z = quaternion.z / norm
        quaternion.w = quaternion.w / norm

    return quaternion