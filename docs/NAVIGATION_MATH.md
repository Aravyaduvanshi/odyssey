# Navigation Math

Phase 1 does not implement strapdown INS or EKF yet.

The current local display frame is ENU:

- X: East
- Y: North
- Z: Up

GNSS latitude/longitude from the synthetic provider is converted to a local tangent approximation for dashboard display only. Future INS/EKF phases must use quaternion orientation propagation, gravity compensation, covariance tracking, and proper measurement updates.

