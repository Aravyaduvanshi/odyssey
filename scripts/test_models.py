#!/usr/bin/env python3
"""
OdysseyV1 Model Smoke Test Script
Validates MountNet V2 and Velocity TCN inference, preprocessing, denormalization, and latencies.
"""

import math
import sys
import numpy as np

from edge.models.manager import ModelManager


def main() -> int:
    print("=============================")
    print("ODYSSEYV1 MODEL TEST")
    print("=============================")

    # Initialize ModelManager (which loads and warms up both models)
    manager = ModelManager(auto_load=True)

    # 1. Test MountNet V2
    statuses = manager.statuses()
    if statuses["yaw"] != "READY":
        print(f"FAIL: MountNet status is {statuses['yaw']}, error: {manager.mountnet.error_message}")
        return 1

    # Create dummy valid 100x6 phone-frame IMU input (e.g. flat phone with gravity along Z)
    phone_window = np.zeros((100, 6), dtype=np.float32)
    phone_window[:, 2] = 9.80665  # az ≈ 1g

    yaw_pred = manager.predict_yaw(phone_window, timestamp_ns=1_000_000_000)
    if yaw_pred is None:
        print(f"FAIL: MountNet prediction returned None. Error: {manager.mountnet.error_message}")
        return 1

    if not math.isfinite(yaw_pred.mounting_yaw_rad) or not math.isfinite(yaw_pred.mounting_yaw_deg):
        print(f"FAIL: Non-finite MountNet output: {yaw_pred}")
        return 1

    print("MountNet:")
    print(f"status: {statuses['yaw']}")
    print("input: (1,100,6)")
    print(f"yaw: {yaw_pred.mounting_yaw_deg:.1f} deg ({yaw_pred.mounting_yaw_rad:.3f} rad)")
    print(f"sin_yaw: {yaw_pred.sin_yaw:.4f}, cos_yaw: {yaw_pred.cos_yaw:.4f}")
    print(f"latency: {yaw_pred.inference_latency_ms:.1f} ms")

    print()

    # 2. Test Velocity TCN
    if statuses["velocity"] != "READY":
        print(f"FAIL: Velocity TCN status is {statuses['velocity']}, error: {manager.velocity.error_message}")
        return 1

    # Create dummy valid 100x6 vehicle-aligned IMU input
    # [ACC_VEH_X, ACC_VEH_Y, ACC_VEH_Z, GYRO_Yaw_corrected, GYRO_Pitch_corrected, GYRO_Roll_corrected]
    vehicle_window = np.zeros((100, 6), dtype=np.float32)
    vehicle_window[:, 0] = 0.0  # forward accel
    vehicle_window[:, 2] = 9.80665  # vertical accel

    vel_pred = manager.predict_velocity(vehicle_window, timestamp_ns=1_000_000_000)
    if vel_pred is None:
        print(f"FAIL: Velocity TCN prediction returned None. Error: {manager.velocity.error_message}")
        return 1

    if not math.isfinite(vel_pred.velocity_mps) or not math.isfinite(vel_pred.log_variance):
        print(f"FAIL: Non-finite Velocity TCN output: {vel_pred}")
        return 1

    print("Velocity TCN:")
    print(f"status: {statuses['velocity']}")
    print("input: (1,100,6)")
    print(f"velocity: {vel_pred.velocity_mps:.1f} m/s")
    print(f"velocity: {vel_pred.velocity_kmh:.1f} km/h")
    print(f"normalized_output: {vel_pred.velocity_normalized:.4f}")
    print(f"log_variance: {vel_pred.log_variance:.4f}")
    if vel_pred.sigma_mps is not None:
        print(f"sigma: {vel_pred.sigma_mps:.3f} m/s")
    print(f"latency: {vel_pred.inference_latency_ms:.1f} ms")

    print("=============================")
    print("PASS")
    print("=============================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
