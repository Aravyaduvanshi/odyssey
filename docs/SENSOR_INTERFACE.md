# Sensor Interface

Every provider must yield `SensorSample` in SI units:

- acceleration: m/s²
- angular velocity: rad/s
- velocity: m/s
- heading: rad

Implemented:

- `MockSensorProvider`

Reserved interfaces:

- `SmartphoneSensorProvider`
- `IOVNBDProvider`
- `CSVProvider`
- `ReplayProvider`
- `ExternalIMUProvider`
- future `FOGIMUProvider`

