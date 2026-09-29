import numpy as np


class InnovationGating:
    """Statistical outlier rejection using Chi-Square Mahalanobis distance."""

    @staticmethod
    def test(
        innovation: np.ndarray,
        innovation_cov: np.ndarray,
        threshold: float,
    ) -> tuple[bool, float]:
        """
        Compute Mahalanobis distance gamma = y^T * S^-1 * y.
        Returns (accepted, distance_squared).
        """
        y = np.asarray(innovation, dtype=float)
        s = np.asarray(innovation_cov, dtype=float)

        try:
            # Solve S * x = y for x = S^-1 * y cleanly using linalg solve
            if s.size == 1:
                s_val = float(s.item())
                y_val = float(y.item())
                if s_val <= 0.0 or not np.isfinite(s_val):
                    return False, float("inf")
                dist_sq = float((y_val ** 2) / s_val)
            else:
                inv_s_y = np.linalg.solve(s, y)
                dist_sq = float(y.T @ inv_s_y)

            if not np.isfinite(dist_sq) or dist_sq < 0.0:
                return False, float("inf")

            accepted = dist_sq <= threshold
            return accepted, dist_sq
        except np.linalg.LinAlgError:
            return False, float("inf")
