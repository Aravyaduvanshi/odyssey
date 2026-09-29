from dataclasses import dataclass
from enum import IntEnum
import numpy as np

# 15-state Error-State layout
STATE_DIM = 15

# Slice indices (for block matrix assignments)
POSITION = slice(0, 3)     # delta_p (East, North, Up) in meters
VELOCITY = slice(3, 6)     # delta_v (v_E, v_N, v_U) in m/s
ATTITUDE = slice(6, 9)     # delta_theta (roll, pitch, yaw error) in radians
ACCEL_BIAS = slice(9, 12)  # delta_b_a (x, y, z bias) in m/s^2
GYRO_BIAS = slice(12, 15)  # delta_b_g (x, y, z bias) in rad/s

# Aliases for backwards compatibility
POS_IDX = POSITION
VEL_IDX = VELOCITY
ATT_IDX = ATTITUDE
ACC_BIAS_IDX = ACCEL_BIAS
GYRO_BIAS_IDX = GYRO_BIAS


class ErrorStateIndex(IntEnum):
    """Exact element indices for the 15-state error vector."""
    POS_X = 0
    POS_Y = 1
    POS_Z = 2
    VEL_X = 3
    VEL_Y = 4
    VEL_Z = 5
    ATT_X = 6
    ATT_Y = 7
    ATT_Z = 8
    ACC_BIAS_X = 9
    ACC_BIAS_Y = 10
    ACC_BIAS_Z = 11
    GYRO_BIAS_X = 12
    GYRO_BIAS_Y = 13
    GYRO_BIAS_Z = 14


def skew(v: np.ndarray) -> np.ndarray:
    """
    Return 3x3 skew-symmetric cross-product matrix [v]_x such that:
        skew(v) @ x == np.cross(v, x)
    for any 3D vector x.
    """
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=float)


# Alias
skew_symmetric = skew
