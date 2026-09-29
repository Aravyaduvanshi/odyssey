from dataclasses import dataclass

import numpy as np

from edge.coordinate.quaternion import Quaternion, from_euler_rad


@dataclass(frozen=True)
class PhoneFrame:
    convention: str = "phone/body frame from sensor provider"


@dataclass(frozen=True)
class VehicleFrame:
    x: str = "forward"
    y: str = "lateral-left"
    z: str = "vertical-up"


@dataclass(frozen=True)
class NavigationFrame:
    convention: str = "ENU: x=east, y=north, z=up"


@dataclass(frozen=True)
class ImuVector:
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


@dataclass(frozen=True)
class VehicleAlignedIMU:
    acc_veh_x: float
    acc_veh_y: float
    acc_veh_z: float
    gyro_yaw_corrected: float
    gyro_pitch_corrected: float
    gyro_roll_corrected: float

    def as_features(self) -> np.ndarray:
        return np.array([
            self.acc_veh_x,
            self.acc_veh_y,
            self.acc_veh_z,
            self.gyro_yaw_corrected,
            self.gyro_pitch_corrected,
            self.gyro_roll_corrected,
        ], dtype=np.float32)

    def as_ai_features(self, gravity_mps2: float = 9.80665) -> np.ndarray:
        """
        Features formatted strictly for the Velocity TCN input contract:
        specific force -> gravity compensation -> linear acceleration -> Velocity TCN.
        Linear vertical acceleration is computed as: acc_veh_z - gravity_mps2.
        For level stationary data, ACC_VEH_Z ≈ 0 m/s^2.
        """
        return np.array([
            self.acc_veh_x,
            self.acc_veh_y,
            self.acc_veh_z - gravity_mps2,
            self.gyro_yaw_corrected,
            self.gyro_pitch_corrected,
            self.gyro_roll_corrected,
        ], dtype=np.float32)

    def as_dict(self) -> dict[str, float]:
        return {
            "ACC_VEH_X": self.acc_veh_x,
            "ACC_VEH_Y": self.acc_veh_y,
            "ACC_VEH_Z": self.acc_veh_z,
            "GYRO_Yaw_corrected": self.gyro_yaw_corrected,
            "GYRO_Pitch_corrected": self.gyro_pitch_corrected,
            "GYRO_Roll_corrected": self.gyro_roll_corrected,
        }


class VehicleFrameTransform:
    def __init__(self, phone_to_vehicle_quaternion: Quaternion | None = None) -> None:
        self.phone_to_vehicle_quaternion = phone_to_vehicle_quaternion or Quaternion(1.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_alignment(cls, roll_rad: float, pitch_rad: float, yaw_rad: float) -> "VehicleFrameTransform":
        return cls(from_euler_rad(roll_rad, pitch_rad, yaw_rad))

    def phone_to_vehicle(self, imu: ImuVector) -> ImuVector:
        accel = self.phone_to_vehicle_quaternion.rotate_vector(np.array([imu.ax, imu.ay, imu.az], dtype=float))
        gyro = self.phone_to_vehicle_quaternion.rotate_vector(np.array([imu.gx, imu.gy, imu.gz], dtype=float))
        return ImuVector(
            ax=float(accel[0]),
            ay=float(accel[1]),
            az=float(accel[2]),
            gx=float(gyro[0]),
            gy=float(gyro[1]),
            gz=float(gyro[2]),
        )

    def to_vehicle_aligned_imu(self, imu: ImuVector) -> VehicleAlignedIMU:
        """
        Transform raw phone IMU to vehicle-aligned features in exact order expected by Velocity TCN:
        ACC_VEH_X (forward), ACC_VEH_Y (lateral), ACC_VEH_Z (vertical),
        GYRO_Yaw_corrected (Z-rate), GYRO_Pitch_corrected (Y-rate), GYRO_Roll_corrected (X-rate).
        """
        accel = self.phone_to_vehicle_quaternion.rotate_vector(np.array([imu.ax, imu.ay, imu.az], dtype=float))
        gyro = self.phone_to_vehicle_quaternion.rotate_vector(np.array([imu.gx, imu.gy, imu.gz], dtype=float))
        return VehicleAlignedIMU(
            acc_veh_x=float(accel[0]),
            acc_veh_y=float(accel[1]),
            acc_veh_z=float(accel[2]),
            gyro_yaw_corrected=float(gyro[2]),
            gyro_pitch_corrected=float(gyro[1]),
            gyro_roll_corrected=float(gyro[0]),
        )

    def vehicle_to_navigation(self, vector: np.ndarray, heading_rad: float) -> np.ndarray:
        heading = from_euler_rad(0.0, 0.0, heading_rad)
        return heading.rotate_vector(vector)

