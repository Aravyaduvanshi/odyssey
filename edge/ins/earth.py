import math
import numpy as np

# WGS-84 Ellipsoid constants
WGS84_A = 6378137.0                  # semi-major axis (meters)
WGS84_F = 1.0 / 298.257223563        # flattening
WGS84_E2 = 2.0 * WGS84_F - WGS84_F ** 2  # first eccentricity squared


class EarthModel:
    """WGS-84 Earth ellipsoid and local ENU coordinate transformations."""

    @staticmethod
    def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
        """Convert geodetic (lat, lon, alt) to Earth-Centered Earth-Fixed (ECEF) Cartesian coordinates."""
        lat_rad = math.radians(lat_deg)
        lon_rad = math.radians(lon_deg)
        sin_lat = math.sin(lat_rad)
        cos_lat = math.cos(lat_rad)
        sin_lon = math.sin(lon_rad)
        cos_lon = math.cos(lon_rad)

        # Prime vertical radius of curvature
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)

        x = (n + alt_m) * cos_lat * cos_lon
        y = (n + alt_m) * cos_lat * sin_lon
        z = (n * (1.0 - WGS84_E2) + alt_m) * sin_lat
        return np.array([x, y, z], dtype=float)

    @staticmethod
    def ecef_to_enu(ecef: np.ndarray, ref_lat_deg: float, ref_lon_deg: float, ref_alt_m: float) -> np.ndarray:
        """Convert ECEF vector to local ENU tangent plane relative to reference point."""
        ref_ecef = EarthModel.geodetic_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m)
        dx = ecef - ref_ecef

        lat_rad = math.radians(ref_lat_deg)
        lon_rad = math.radians(ref_lon_deg)
        sin_lat = math.sin(lat_rad)
        cos_lat = math.cos(lat_rad)
        sin_lon = math.sin(lon_rad)
        cos_lon = math.cos(lon_rad)

        r_enu_ecef = np.array([
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ], dtype=float)

        return r_enu_ecef @ dx

    @staticmethod
    def geodetic_to_enu(
        lat_deg: float, lon_deg: float, alt_m: float,
        ref_lat_deg: float, ref_lon_deg: float, ref_alt_m: float
    ) -> np.ndarray:
        """Direct geodetic (lat, lon, alt) to local ENU tangent coordinates in meters."""
        ecef = EarthModel.geodetic_to_ecef(lat_deg, lon_deg, alt_m)
        return EarthModel.ecef_to_enu(ecef, ref_lat_deg, ref_lon_deg, ref_alt_m)

    @staticmethod
    def enu_to_geodetic(
        enu: np.ndarray,
        ref_lat_deg: float, ref_lon_deg: float, ref_alt_m: float
    ) -> tuple[float, float, float]:
        """Convert local ENU coordinates back to geodetic (lat, lon, alt) using Bowring's method."""
        lat_rad = math.radians(ref_lat_deg)
        lon_rad = math.radians(ref_lon_deg)
        sin_lat = math.sin(lat_rad)
        cos_lat = math.cos(lat_rad)
        sin_lon = math.sin(lon_rad)
        cos_lon = math.cos(lon_rad)

        # Inverse rotation: ECEF to ENU transpose
        r_ecef_enu = np.array([
            [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
            [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
            [0.0, cos_lat, sin_lat],
        ], dtype=float)

        ref_ecef = EarthModel.geodetic_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m)
        ecef = ref_ecef + r_ecef_enu @ enu

        # Bowring's closed-form algorithm
        x, y, z = ecef[0], ecef[1], ecef[2]
        p = math.sqrt(x * x + y * y)
        b = WGS84_A * (1.0 - WGS84_F)
        e_prime2 = (WGS84_A * WGS84_A - b * b) / (b * b)

        theta = math.atan2(z * WGS84_A, p * b)
        lat = math.atan2(
            z + e_prime2 * b * (math.sin(theta) ** 3),
            p - WGS84_E2 * WGS84_A * (math.cos(theta) ** 3)
        )
        lon = math.atan2(y, x)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * (math.sin(lat) ** 2))
        alt = p / math.cos(lat) - n

        return math.degrees(lat), math.degrees(lon), alt
