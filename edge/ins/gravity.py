import numpy as np

DEFAULT_GRAVITY_MAGNITUDE = 9.80665  # m/s^2


class GravityModel:
    """
    Local ENU Gravity Model.
    In the East-North-Up (ENU) convention:
    Gravity acts downward towards Earth's center: g_nav = [0.0, 0.0, -g]^T.
    
    When stationary at rest with vertical axis pointing UP:
    Accelerometer measures specific force: f_nav = [0.0, 0.0, +g]^T (reaction to gravity).
    Navigation acceleration: a_nav = f_nav + g_nav = [0.0, 0.0, 0.0]^T.
    """

    def __init__(self, magnitude: float = DEFAULT_GRAVITY_MAGNITUDE) -> None:
        if magnitude <= 0.0:
            raise ValueError("Gravity magnitude must be positive.")
        self.magnitude = float(magnitude)

    def vector_enu(self) -> np.ndarray:
        """Return [0.0, 0.0, -magnitude] in local ENU frame."""
        return np.array([0.0, 0.0, -self.magnitude], dtype=float)

    @classmethod
    def normal_gravity_somigliana(cls, latitude_deg: float, altitude_m: float = 0.0) -> "GravityModel":
        """
        Somigliana equation for WGS-84 normal gravity as a function of latitude and altitude.
        gamma_0 = 9.7803253359 * (1 + 0.0053024 * sin^2(lat) - 0.0000058 * sin^2(2*lat))
        gamma = gamma_0 - (3.086e-6 * h)
        """
        lat_rad = np.radians(latitude_deg)
        sin_lat = np.sin(lat_rad)
        sin_2lat = np.sin(2.0 * lat_rad)
        gamma_0 = 9.7803253359 * (1.0 + 0.0053024 * (sin_lat ** 2) - 0.0000058 * (sin_2lat ** 2))
        g = gamma_0 - 3.086e-6 * altitude_m
        return cls(g)
