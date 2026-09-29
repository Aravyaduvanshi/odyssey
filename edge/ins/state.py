from dataclasses import dataclass
import numpy as np

from edge.coordinate.quaternion import Quaternion, to_euler_rad


@dataclass
class INSState:
    """
    Nominal Strapdown INS State.
    
    Frames:
    - Position: 3D Local Tangent ENU [East, North, Up] (meters)
    - Velocity: 3D Local Tangent ENU [v_E, v_N, v_U] (m/s)
    - Orientation: Quaternion q_n_v mapping vectors from Vehicle frame to Navigation (ENU) frame
    - Accel Bias: 3D vehicle-frame accelerometer bias [b_ax, b_ay, b_az] (m/s^2)
    - Gyro Bias: 3D vehicle-frame gyroscope bias [b_gx, b_gy, b_gz] (rad/s)
    - Timestamp: Nanoseconds
    """
    position_enu: np.ndarray       # shape (3,), float
    velocity_enu: np.ndarray       # shape (3,), float
    quaternion: Quaternion         # unit quaternion q_n_v
    accel_bias: np.ndarray         # shape (3,), float (m/s^2)
    gyro_bias: np.ndarray          # shape (3,), float (rad/s)
    timestamp_ns: int = 0

    def __post_init__(self) -> None:
        self.position_enu = np.asarray(self.position_enu, dtype=float).reshape(3)
        self.velocity_enu = np.asarray(self.velocity_enu, dtype=float).reshape(3)
        self.accel_bias = np.asarray(self.accel_bias, dtype=float).reshape(3)
        self.gyro_bias = np.asarray(self.gyro_bias, dtype=float).reshape(3)
        self.quaternion = self.quaternion.normalized()

    @classmethod
    def zero(cls, timestamp_ns: int = 0) -> "INSState":
        return cls(
            position_enu=np.zeros(3, dtype=float),
            velocity_enu=np.zeros(3, dtype=float),
            quaternion=Quaternion(1.0, 0.0, 0.0, 0.0),
            accel_bias=np.zeros(3, dtype=float),
            gyro_bias=np.zeros(3, dtype=float),
            timestamp_ns=timestamp_ns,
        )

    def copy(self) -> "INSState":
        return INSState(
            position_enu=self.position_enu.copy(),
            velocity_enu=self.velocity_enu.copy(),
            quaternion=Quaternion(self.quaternion.w, self.quaternion.x, self.quaternion.y, self.quaternion.z),
            accel_bias=self.accel_bias.copy(),
            gyro_bias=self.gyro_bias.copy(),
            timestamp_ns=self.timestamp_ns,
        )

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Direction Cosine Matrix R_n_v (3x3) rotating a vector from Vehicle frame to Navigation (ENU) frame:
        v_nav = R_n_v @ v_veh
        """
        q = self.quaternion
        w, x, y, z = q.w, q.x, q.y, q.z
        return np.array([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ], dtype=float)

    @property
    def euler_rad(self) -> tuple[float, float, float]:
        """Return (roll, pitch, yaw) in radians in navigation frame."""
        return to_euler_rad(self.quaternion)

    def to_dict(self) -> dict[str, object]:
        roll, pitch, yaw = self.euler_rad
        return {
            "position_enu": self.position_enu.tolist(),
            "velocity_enu": self.velocity_enu.tolist(),
            "quaternion": [self.quaternion.w, self.quaternion.x, self.quaternion.y, self.quaternion.z],
            "roll_rad": roll,
            "pitch_rad": pitch,
            "yaw_rad": yaw,
            "accel_bias": self.accel_bias.tolist(),
            "gyro_bias": self.gyro_bias.tolist(),
            "timestamp_ns": self.timestamp_ns,
        }
