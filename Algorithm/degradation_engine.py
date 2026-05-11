import numpy as np


class DegradationEngine:
    """Extreme physical-degradation pipeline for Sim2Real domain randomization.

    Applies four categories of corruption to a clean point cloud:
      1. Sensor noise          — Gaussian speckle on XYZ
      2. Z-density decay       — LiDAR-like dropout, sparser at range
      3. Occlusion holes       — random spherical cut-outs (vegetation)
      4. Random tilt           — slight 3D rotation (construction deviation)

    All operations are vectorised over points.  The ``degrade()`` convenience
    method chains the enabled steps; point-removing steps automatically
    filter *labels* as well.
    """

    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Each value is a (lo, hi) uniform range.
        # Set any flag to False to skip that degradation.
        self.cfg = {
            # ---- sensor noise ----
            "noise_std_lo": 0.01,   # 1 cm
            "noise_std_hi": 0.05,   # 5 cm
            "enable_noise": True,
            # ---- Z-density decay ----
            "decay_max_dropout": (0.15, 0.55),  # max drop probability at z_max
            "enable_decay": True,
            # ---- occlusion holes ----
            "occlusion_n": (1, 8),
            "occlusion_r": (0.2, 2.0),  # metres
            "enable_occlusion": True,
            # ---- random tilt ----
            "tilt_deg": (0.0, 3.0),  # degrees
            "enable_tilt": True,
        }

    # ------------------------------------------------------------------
    #  helpers
    # ------------------------------------------------------------------

    def _r(self, key):
        lo, hi = self.cfg[key]
        return np.random.uniform(lo, hi)

    @staticmethod
    def _rotation_matrix(axis, angle_rad):
        """Rodrigues formula: return 3×3 rotation matrix."""
        axis = np.asarray(axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], dtype=np.float64)
        return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)

    # ------------------------------------------------------------------
    #  degradation primitives
    # ------------------------------------------------------------------

    def apply_sensor_noise(self, points):
        """Add isotropic Gaussian noise to XYZ coordinates.

        Returns
        -------
        points : (N, 3)  — same count, perturbed.
        """
        if not self.cfg["enable_noise"]:
            return points
        std = np.random.uniform(self.cfg["noise_std_lo"], self.cfg["noise_std_hi"])
        return points + np.random.normal(0, std, points.shape).astype(points.dtype)

    def apply_z_density_decay(self, points, labels):
        """Drop points with probability proportional to their Z coordinate.

        Higher points = sparser returns (LiDAR range effect).

        Returns
        -------
        points, labels  — filtered.
        """
        if not self.cfg["enable_decay"]:
            return points, labels
        z = points[:, 2]
        z_min, z_max = z.min(), z.max()
        if z_max - z_min < 1e-6:
            return points, labels
        max_drop = self._r("decay_max_dropout")
        drop_prob = max_drop * (z - z_min) / (z_max - z_min)
        keep = np.random.uniform(0, 1, len(points)) >= drop_prob
        if labels is not None:
            labels = labels[keep]
        return points[keep], labels

    def apply_occlusion_holes(self, points, labels):
        """Delete points that fall inside randomly placed spheres.

        Simulates severe vegetation occlusion / beam blockage.

        Returns
        -------
        points, labels  — filtered.
        """
        if not self.cfg["enable_occlusion"]:
            return points, labels
        n = int(np.random.randint(*self.cfg["occlusion_n"]))
        if n == 0:
            return points, labels

        r = self._r("occlusion_r")
        lo = points.min(axis=0)
        hi = points.max(axis=0)
        span = hi - lo

        # random sphere centres inside point-cloud bounding box
        centres = lo[None, :] + np.random.uniform(0, 1, (n, 3)) * span[None, :]
        r_arr = r * np.ones(n)  # same radius for all holes (could randomise each)

        # vectorised per-point vs per-sphere distance check — (N, M)
        # avoid O(N·M) memory blow-out for large N by looping over spheres
        keep = np.ones(len(points), dtype=bool)
        for i in range(n):
            d2 = np.sum((points - centres[i]) ** 2, axis=1)
            keep &= (d2 >= r_arr[i] ** 2)

        if labels is not None:
            labels = labels[keep]
        return points[keep], labels

    def apply_random_tilt(self, points):
        """Apply a small random 3D rotation (simulates construction deviation)."""
        if not self.cfg["enable_tilt"]:
            return points
        deg = self._r("tilt_deg")
        # random axis — uniform on sphere
        axis = np.random.normal(0, 1, 3).astype(np.float64)
        axis /= np.linalg.norm(axis)
        R = self._rotation_matrix(axis, np.deg2rad(deg))
        return (points @ R.T).astype(points.dtype)

    # ------------------------------------------------------------------
    #  convenience pipeline
    # ------------------------------------------------------------------

    def degrade(self, points, labels):
        """Run all enabled degradations in a sensible order.

        Order:
          1. noise       (perturb positions, count unchanged)
          2. tilt        (rotate, count unchanged)
          3. decay       (drop by Z, count reduced)
          4. occlusion   (spherical cut-outs, count reduced)

        Parameters
        ----------
        points : (N, 3)  float32/float64
        labels : (N, 1)  or None

        Returns
        -------
        points : (N', 3)
        labels : (N', 1) or None
        """
        points = self.apply_sensor_noise(points)
        points = self.apply_random_tilt(points)   # tilt before removal = cheaper
        points, labels = self.apply_z_density_decay(points, labels)
        points, labels = self.apply_occlusion_holes(points, labels)
        return points, labels
