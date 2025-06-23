import numpy as np
from geometry_msgs.msg import *

'''
Credit to https://github.com/iamlucaswolf for these functions which I slightly simplified
'''


stamped_type_to_attr = {
    TwistStamped: 'twist',
    TwistWithCovarianceStamped: 'twist',
    Vector3Stamped: 'vector',
    WrenchStamped: 'wrench',
}

def cast_to_dtype(array, dtype):
    """Raises a TypeError if `array` cannot be casted to `dtype` without
    loss of precision."""

    min_dtype = np.min_scalar_type(array)

    if not np.can_cast(min_dtype, dtype):
        raise TypeError(f'Cannot safely cast array {array} to dtype {dtype}.')

    return array.astype(dtype)

def unstamp(message):
    attribute = stamped_type_to_attr.get(message.__class__)    
    if attribute is not None:
        message = getattr(message, attribute)
        
    return message

def kinematics_to_numpy(message):
    message = unstamp(message)
    return vector_to_numpy(message.linear), vector_to_numpy(message.angular)

def kinematics_with_covariance_to_numpy(message):
    message = unstamp(message)
    is_accel = isinstance(message, AccelWithCovariance)
    
    kinematics = message.accel if is_accel else message.twist
    
    linear, angular = kinematics_to_numpy(kinematics)
    covariance = np.array(message.covariance, dtype=np.float64).reshape(6, 6)
    
    return linear, angular, covariance

def numpy_to_vector(message_type, array):

    dtype = np.float32 if message_type is Point32 else np.float64
    array = cast_to_dtype(array, dtype)

    return message_type(*array[:3])

def vector_to_numpy(vec):
    '''
    Converts a ros vector type to a numpy array
    '''
    vec = unstamp(vec)
    data = [vec.x, vec.y, vec.z]
    d_type = np.float32 if isinstance(vec, Point32) else np.float64
    
    return np.array(data, dtype=d_type)

def numpy_to_kinematics(message_type, linear, angular):
    linear_key = 'linear'
    angular_key = 'angular'

    kwargs = {
        linear_key: numpy_to_vector(Vector3, linear),
        angular_key: numpy_to_vector(Vector3, angular)
    }

    return message_type(**kwargs)