import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Quaternion:
    w: float
    x: float
    y: float
    z: float

    def as_array(self) -> np.ndarray:
        return np.array([self.w, self.x, self.y, self.z], dtype=float)

    def normalized(self) -> "Quaternion":
        norm = float(np.linalg.norm(self.as_array()))
        if norm <= 0.0 or not math.isfinite(norm):
            raise ValueError("invalid quaternion norm")
        return Quaternion(self.w / norm, self.x / norm, self.y / norm, self.z / norm)

    def inverse(self) -> "Quaternion":
        q = self.normalized()
        return Quaternion(q.w, -q.x, -q.y, -q.z)

    def multiply(self, other: "Quaternion") -> "Quaternion":
        a = self.normalized()
        b = other.normalized()
        return _hamilton(a, b).normalized()

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        return self.multiply(other)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """3x3 rotation matrix (DCM) mapping vector from vehicle body to navigation frame."""
        q = self.normalized()
        w, x, y, z = q.w, q.x, q.y, q.z
        return np.array([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ], dtype=float)

    def rotate_vector(self, vector: np.ndarray) -> np.ndarray:
        q = self.normalized()
        qv = Quaternion(0.0, float(vector[0]), float(vector[1]), float(vector[2]))
        rotated = _hamilton(_hamilton(q, qv), q.inverse())
        return np.array([rotated.x, rotated.y, rotated.z], dtype=float)

    def to_euler_rad(self) -> tuple[float, float, float]:
        return to_euler_rad(self)

    @classmethod
    def from_rotation_vector(cls, rotation_vector: np.ndarray) -> "Quaternion":
        return from_rotation_vector(rotation_vector)

    @classmethod
    def from_euler_rad(cls, roll: float, pitch: float, yaw: float) -> "Quaternion":
        return from_euler_rad(roll, pitch, yaw)


def _hamilton(a: Quaternion, b: Quaternion) -> Quaternion:
    return Quaternion(
            a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
            a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    )


def from_euler_rad(roll: float, pitch: float, yaw: float) -> Quaternion:
    cr = math.cos(roll / 2.0)
    sr = math.sin(roll / 2.0)
    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)
    cy = math.cos(yaw / 2.0)
    sy = math.sin(yaw / 2.0)
    return Quaternion(
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ).normalized()


def to_euler_rad(q: Quaternion) -> tuple[float, float, float]:
    n = q.normalized()
    sinr_cosp = 2.0 * (n.w * n.x + n.y * n.z)
    cosr_cosp = 1.0 - 2.0 * (n.x * n.x + n.y * n.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (n.w * n.y - n.z * n.x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (n.w * n.z + n.x * n.y)
    cosy_cosp = 1.0 - 2.0 * (n.y * n.y + n.z * n.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def from_rotation_vector(rotation_vector: np.ndarray) -> Quaternion:
    angle = float(np.linalg.norm(rotation_vector))
    if angle <= 0.0:
        return Quaternion(1.0, 0.0, 0.0, 0.0)
    axis = rotation_vector / angle
    half = angle / 2.0
    return Quaternion(
        math.cos(half),
        float(axis[0]) * math.sin(half),
        float(axis[1]) * math.sin(half),
        float(axis[2]) * math.sin(half),
    ).normalized()
