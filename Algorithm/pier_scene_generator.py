import numpy as np

from degradation_engine import DegradationEngine


class PierSceneGenerator:
    """Programmatic generator for bridge pier point cloud scenes.

    Generates synthetic point clouds for Sim2Real training of pier segmentation models.
    Three pier morphologies are supported:
      (a) Standard cylindrical pier
      (b) Rectangular/trapezoidal gravity-style solid pier
      (c) Y-shaped pier with bifurcated branches

    Each scene includes engineering context (deck + ground) and hard negative samples
    (slender cylinders simulating street-lights / scaffolding).
    """

    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.cfg = {
            # ---- Cylindrical pier ----
            "cyl_radius": (0.5, 1.5),
            "cyl_height": (6.0, 15.0),
            # ---- Gravity pier (truncated rectangular pyramid) ----
            "grav_bw": (1.5, 3.5),  # bottom width
            "grav_bd": (1.0, 3.0),  # bottom depth
            "grav_tw": (1.0, 2.5),  # top width
            "grav_td": (0.7, 2.0),  # top depth
            "grav_h": (5.0, 12.0),
            # ---- Y-shaped pier ----
            "y_radius": (0.5, 1.2),
            "y_height": (6.0, 14.0),
            "y_branch_angle": (15, 35),  # degrees from vertical
            "y_branch_len_ratio": (0.3, 0.5),  # branch length fraction of total height
            "y_branch_r_ratio": (0.6, 0.85),  # branch radius fraction of main radius
            # ---- Deck (bridge girder) ----
            "deck_w_factor": (2.5, 4.5),  # multiplier on pier width
            "deck_len": (4.0, 10.0),
            "deck_thick": (0.4, 1.2),
            # ---- Ground / water surface ----
            "gnd_half": (12.0, 28.0),
            "gnd_z_off": (-0.3, 0.0),
            # ---- Hard negatives (street-lights / scaffolding) ----
            "neg_n": (2, 6),
            "neg_r": (0.05, 0.18),
            "neg_h": (3.0, 9.0),
            "neg_dist": (2.5, 9.0),
            # ---- Realistic hard negatives (look-like-pier structures) ----
            "hard_tree_n": (1, 3),        # 树干
            "hard_tree_r": (0.3, 0.8),    # 树干半径 (和桥墩重叠!)
            "hard_tree_h": (4.0, 12.0),
            "hard_pole_n": (1, 3),        # 电线杆
            "hard_pole_r": (0.08, 0.2),
            "hard_pole_h": (6.0, 15.0),
            "hard_column_n": (1, 2),      # 建筑立柱 (和重力墩几何重叠!)
            "hard_column_w": (0.4, 1.2),
            "hard_column_d": (0.4, 1.2),
            "hard_column_h": (3.0, 8.0),
            "hard_brace_n": (1, 2),       # 斜撑
            "hard_brace_r": (0.15, 0.4),
            "hard_brace_len": (3.0, 8.0),
            "hard_brace_angle": (25, 60),  # degrees from vertical
            # ---- Point budgets ----
            "n_pier": 3000,
            "n_deck": 1500,
            "n_gnd": 1000,
            "n_neg": 300,  # per negative instance
            "n_hard_neg": 500,  # per hard negative instance
            # ---- Misc ----
            "noise": 0.01,
        }

        self.degrader = DegradationEngine(seed=seed)

    # ------------------------------------------------------------------
    #  helpers
    # ------------------------------------------------------------------

    def _r(self, key):
        """Sample uniformly from a (lo, hi) range in *cfg*."""
        lo, hi = self.cfg[key]
        return np.random.uniform(lo, hi)

    @staticmethod
    def _add_noise(pts, sigma):
        if sigma <= 0:
            return pts
        return pts + np.random.normal(0, sigma, pts.shape).astype(pts.dtype)

    # ------------------------------------------------------------------
    #  surface samplers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_cylinder(radius, height, n, base_z=0.0):
        """Points on the surface of a vertical cylinder (lateral + two caps).

        Points are allocated proportionally to surface area.
        """
        area_lat = 2 * np.pi * radius * height
        area_cap = np.pi * radius * radius
        total = area_lat + 2 * area_cap

        n_lat = max(1, int(np.round(n * area_lat / total)))
        n_cap = max(1, int(np.round(n * area_cap / total)))

        # ---- lateral surface ----
        theta = np.random.uniform(0, 2 * np.pi, n_lat)
        z = np.random.uniform(0, height, n_lat)
        lat = np.column_stack([radius * np.cos(theta),
                               radius * np.sin(theta),
                               base_z + z])

        # ---- caps (disk sampling with sqrt for uniform fill) ----
        for cap_z, count in ((base_z, n_cap), (base_z + height, n_cap)):
            r = radius * np.sqrt(np.random.uniform(0, 1, count))
            th = np.random.uniform(0, 2 * np.pi, count)
            cap = np.column_stack([r * np.cos(th),
                                    r * np.sin(th),
                                    np.full(count, cap_z)])
            lat = np.vstack([lat, cap])

        return lat

    @staticmethod
    def _sample_tilted_cylinder(start, direction, radius, length, n):
        """Points on the surface of a tilted cylinder (lateral + two caps).

        Parameters
        ----------
        start : (3,) array
            Center of the bottom cap.
        direction : (3,) array
            Unit vector along the cylinder axis.
        radius, length : float
        n : int
            Approximate number of points.

        Returns
        -------
        pts : (N, 3) array
        """
        d = np.asarray(direction, dtype=np.float64)
        d = d / np.linalg.norm(d)

        # orthonormal frame
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if np.abs(np.dot(d, ref)) > 0.999:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        e1 = np.cross(d, ref)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(d, e1)

        area_lat = 2 * np.pi * radius * length
        area_cap = np.pi * radius * radius
        total = area_lat + 2 * area_cap

        n_lat = max(1, int(np.round(n * area_lat / total)))
        n_cap = max(1, int(np.round(n * area_cap / total)))

        # ---- lateral surface ----
        t = np.random.uniform(0, length, n_lat)
        phi = np.random.uniform(0, 2 * np.pi, n_lat)
        lat = (start[None, :]
               + t[:, None] * d[None, :]
               + radius * np.cos(phi)[:, None] * e1[None, :]
               + radius * np.sin(phi)[:, None] * e2[None, :])

        # ---- caps ----
        for cap_t, cnt in ((0.0, n_cap), (length, n_cap)):
            r = radius * np.sqrt(np.random.uniform(0, 1, cnt))
            ph = np.random.uniform(0, 2 * np.pi, cnt)
            cap = (start[None, :]
                   + cap_t * d[None, :]
                   + r[:, None] * np.cos(ph)[:, None] * e1[None, :]
                   + r[:, None] * np.sin(ph)[:, None] * e2[None, :])
            lat = np.vstack([lat, cap])

        return lat

    @staticmethod
    def _sample_truncated_pyramid(bw, bd, tw, td, h, n, base_z=0.0):
        """Points on the surface of a truncated rectangular pyramid.

        Centered at origin.  Bottom rectangle = bw × bd, top = tw × td, height = h.
        """
        # half-sizes
        bhw, bhd = bw / 2, bd / 2
        thw, thd = tw / 2, td / 2

        # area of each trapezoidal face: average edge × h
        area_front  = (bw + tw) / 2 * h  # ±y faces (width edges)
        area_side   = (bd + td) / 2 * h  # ±x faces (depth edges)
        area_top    = tw * td
        area_bottom = bw * bd
        total = 2 * area_front + 2 * area_side + area_top + area_bottom

        n_front  = max(1, int(np.round(n * area_front / total)))
        n_side   = max(1, int(np.round(n * area_side / total)))
        n_top    = max(1, int(np.round(n * area_top / total)))
        n_bottom = max(1, int(np.round(n * area_bottom / total)))

        pieces = []

        # helper — returns half-width and half-depth at height z
        def hw(z):
            return bhw + (thw - bhw) * z / h

        def hd(z):
            return bhd + (thd - bhd) * z / h

        # front / back  (normals ±y)
        for sign in (+1, -1):
            z = np.random.uniform(0, h, n_front)
            w_at_z = hw(z)
            x = np.random.uniform(-w_at_z, w_at_z, n_front)
            y = sign * hd(z)
            pieces.append(np.column_stack([x, y, base_z + z]))

        # left / right  (normals ±x)
        for sign in (+1, -1):
            z = np.random.uniform(0, h, n_side)
            d_at_z = hd(z)
            y = np.random.uniform(-d_at_z, d_at_z, n_side)
            x = sign * hw(z)
            pieces.append(np.column_stack([x, y, base_z + z]))

        # top cap
        x = np.random.uniform(-thw, thw, n_top)
        y = np.random.uniform(-thd, thd, n_top)
        pieces.append(np.column_stack([x, y, np.full(n_top, base_z + h)]))

        # bottom cap
        x = np.random.uniform(-bhw, bhw, n_bottom)
        y = np.random.uniform(-bhd, bhd, n_bottom)
        pieces.append(np.column_stack([x, y, np.full(n_bottom, base_z)]))

        return np.vstack(pieces)

    @staticmethod
    def _sample_box(w, d, h, n, base_z=0.0):
        """Points on the surface of an axis-aligned box, area-proportional."""
        # areas of unique face-pairs
        a_fb = w * d  # front/back  (in yz plane — wait, no. A face's area uses TWO dimensions)
        # Let me re-derive:
        # Top/bottom:  w × d   → each has area w*d
        # Front/back:  w × h   → each has area w*h
        # Left/right:  d × h   → each has area d*h
        area_top_bot = w * d
        area_fb = w * h
        area_lr = d * h
        total = 2 * (area_top_bot + area_fb + area_lr)

        n_tb = max(1, int(np.round(n * area_top_bot / total)))
        n_fb = max(1, int(np.round(n * area_fb / total)))
        n_lr = max(1, int(np.round(n * area_lr / total)))

        pieces = []
        hw, hd = w / 2, d / 2

        # top / bottom
        for z_off in (0.0, h):
            x = np.random.uniform(-hw, hw, n_tb)
            y = np.random.uniform(-hd, hd, n_tb)
            pieces.append(np.column_stack([x, y, np.full(n_tb, base_z + z_off)]))

        # front / back  (±y)
        for y_sign in (+1, -1):
            x = np.random.uniform(-hw, hw, n_fb)
            z = np.random.uniform(0, h, n_fb)
            pieces.append(np.column_stack([x, np.full(n_fb, y_sign * hd), base_z + z]))

        # left / right  (±x)
        for x_sign in (+1, -1):
            y = np.random.uniform(-hd, hd, n_lr)
            z = np.random.uniform(0, h, n_lr)
            pieces.append(np.column_stack([np.full(n_lr, x_sign * hw), y, base_z + z]))

        return np.vstack(pieces)

    @staticmethod
    def _sample_plane(half_w, half_l, z, n):
        """Uniformly sampled points on a horizontal rectangular plane at height z."""
        x = np.random.uniform(-half_w, half_w, n)
        y = np.random.uniform(-half_l, half_l, n)
        return np.column_stack([x, y, np.full(n, z)])

    # ------------------------------------------------------------------
    #  pier generators  (Label = 1)
    # ------------------------------------------------------------------

    def generate_cylinder_pier(self):
        """(a) Standard cylindrical pier."""
        r = self._r("cyl_radius")
        h = self._r("cyl_height")
        n = self.cfg["n_pier"]
        pts = self._sample_cylinder(r, h, n, base_z=0.0)
        pts = self._add_noise(pts, self.cfg["noise"])
        meta = {"type": "cylinder", "radius": r, "height": h,
                "top_z": h, "top_radius": r, "pier_width": 2 * r}
        return pts, meta

    def generate_gravity_pier(self):
        """(b) Rectangular / trapezoidal gravity pier (base >= top)."""
        bw = self._r("grav_bw")
        bd = self._r("grav_bd")
        tw = min(self._r("grav_tw"), bw)
        td = min(self._r("grav_td"), bd)
        h  = self._r("grav_h")
        n = self.cfg["n_pier"]
        pts = self._sample_truncated_pyramid(bw, bd, tw, td, h, n, base_z=0.0)
        pts = self._add_noise(pts, self.cfg["noise"])
        meta = {"type": "gravity", "bottom_w": bw, "bottom_d": bd,
                "top_w": tw, "top_d": td, "height": h,
                "top_z": h, "pier_width": max(bw, tw)}
        return pts, meta

    def generate_y_pier(self):
        """(c) Y-shaped pier with two diverging branches."""
        r_main = self._r("y_radius")
        h_total = self._r("y_height")
        angle_deg = self._r("y_branch_angle")
        angle_rad = np.deg2rad(angle_deg)
        branch_len_ratio = self._r("y_branch_len_ratio")
        branch_r_ratio = self._r("y_branch_r_ratio")

        branch_len = h_total * branch_len_ratio
        h_split = h_total - branch_len * np.cos(angle_rad)
        r_branch = r_main * branch_r_ratio
        n = self.cfg["n_pier"]

        # point budgets: split proportionally to surface area
        area_main = 2 * np.pi * r_main * h_split + 2 * np.pi * r_main ** 2
        area_one_branch = (2 * np.pi * r_branch * branch_len
                           + 2 * np.pi * r_branch ** 2)
        area_total = area_main + 2 * area_one_branch

        n_main = max(1, int(np.round(n * area_main / area_total)))
        n_branch = max(1, int(np.round(n * area_one_branch / area_total)))

        # ---- main column ----
        main_pts = self._sample_cylinder(r_main, h_split, n_main, base_z=0.0)

        # ---- two branches (opposite in x-direction) ----
        start = np.array([0.0, 0.0, h_split])
        branch_pts = []
        for sign in (+1, -1):
            d = np.array([sign * np.sin(angle_rad), 0.0, np.cos(angle_rad)])
            d = d / np.linalg.norm(d)
            bpts = self._sample_tilted_cylinder(start, d, r_branch,
                                                 branch_len, n_branch)
            branch_pts.append(bpts)

        pts = np.vstack([main_pts] + branch_pts)

        # compute pier envelope for downstream use
        max_xy = r_main + branch_len * np.sin(angle_rad)
        top_z = h_split + branch_len * np.cos(angle_rad)

        pts = self._add_noise(pts, self.cfg["noise"])
        meta = {"type": "y_pier", "main_radius": r_main, "height": h_total,
                "branch_angle_deg": angle_deg, "branch_len": branch_len,
                "branch_radius": r_branch, "split_z": h_split,
                "top_z": top_z, "pier_width": 2 * max_xy}
        return pts, meta

    # ------------------------------------------------------------------
    #  context & negatives  (Label = 0)
    # ------------------------------------------------------------------

    def generate_deck(self, pier_top_z, pier_width):
        """Bridge deck: a wide, flat box sitting on top of the pier."""
        w = pier_width * self._r("deck_w_factor")
        length = self._r("deck_len")
        thick = self._r("deck_thick")
        n = self.cfg["n_deck"]

        pts = self._sample_box(w, length, thick, n,
                               base_z=pier_top_z)
        pts = self._add_noise(pts, self.cfg["noise"])
        meta = {"type": "deck", "width": w, "length": length,
                "thickness": thick, "bottom_z": pier_top_z}
        return pts, meta

    def generate_ground(self, pier_base_z=0.0):
        """Horizontal ground / water surface near the pier base."""
        half = self._r("gnd_half")
        z_off = self._r("gnd_z_off")
        n = self.cfg["n_gnd"]
        z = pier_base_z + z_off
        pts = self._sample_plane(half, half, z, n)
        pts = self._add_noise(pts, self.cfg["noise"])
        meta = {"type": "ground", "half_size": half, "z": z}
        return pts, meta

    def generate_negatives(self, pier_envelope_radius, pier_top_z):
        """Hard negatives: slender vertical cylinders placed around the pier.

        Each negative is placed at a random azimuth and distance so that it
        lies outside the pier envelope.
        """
        n_neg = int(np.random.randint(*self.cfg["neg_n"]))
        if n_neg == 0:
            return np.empty((0, 3)), {"type": "negatives", "instances": []}

        pieces = []
        for _ in range(n_neg):
            r = self._r("neg_r")
            h = self._r("neg_h")
            dist = self._r("neg_dist")
            azimuth = np.random.uniform(0, 2 * np.pi)

            actual_dist = pier_envelope_radius + dist + r
            cx = actual_dist * np.cos(azimuth)
            cy = actual_dist * np.sin(azimuth)

            # sample near ground level (base at z ≈ 0 or slightly below)
            base_z = np.random.uniform(-0.5, 0.5)

            n_pts = self.cfg["n_neg"]
            cyl = self._sample_cylinder(r, h, n_pts, base_z=base_z)
            cyl[:, 0] += cx
            cyl[:, 1] += cy
            cyl = self._add_noise(cyl, self.cfg["noise"])
            pieces.append(cyl)

        pts = np.vstack(pieces)
        meta = {"type": "negatives", "n_instances": n_neg}
        return pts, meta

    # ------------------------------------------------------------------
    #  realistic hard negatives (Phase 2 improvement)
    # ------------------------------------------------------------------

    def generate_hard_negatives(self, pier_envelope_radius, pier_top_z):
        """Generate structures that look like piers but aren't.

        Four types:
          1. Tree trunks: thick cylinders, slightly tilted
          2. Utility poles: thin cylinders, possibly with crossbars
          3. Building columns: rectangular prisms (hardest — identical to gravity piers)
          4. Diagonal braces: tilted cylinders
        """
        pieces = []

        # 1. Tree trunks — thick, slightly irregular cylinders
        n_trees = int(np.random.randint(*self.cfg["hard_tree_n"]))
        for _ in range(n_trees):
            r = self._r("hard_tree_r")
            h = self._r("hard_tree_h")
            dist = self._r("neg_dist")
            azimuth = np.random.uniform(0, 2 * np.pi)
            actual_dist = pier_envelope_radius + dist + r
            cx = actual_dist * np.cos(azimuth)
            cy = actual_dist * np.sin(azimuth)
            base_z = np.random.uniform(-0.5, 0.5)
            cyl = self._sample_cylinder(r, h, self.cfg["n_hard_neg"], base_z=base_z)
            cyl[:, 0] += cx
            cyl[:, 1] += cy
            # Slight random tilt for organic look
            tilt_angle = np.random.uniform(0, 5) * np.pi / 180
            tilt_axis = np.random.uniform(0, 2 * np.pi)
            rot_x = np.array([[1, 0, 0],
                              [0, np.cos(tilt_angle), -np.sin(tilt_angle)],
                              [0, np.sin(tilt_angle), np.cos(tilt_angle)]])
            rot_z = np.array([[np.cos(tilt_axis), -np.sin(tilt_axis), 0],
                              [np.sin(tilt_axis), np.cos(tilt_axis), 0],
                              [0, 0, 1]])
            cyl = cyl @ rot_x.T @ rot_z.T
            cyl[:, 0] += cx; cyl[:, 1] += cy
            cyl[:, 2] += base_z
            cyl = self._add_noise(cyl, self.cfg["noise"] * 2)
            pieces.append(cyl)

        # 2. Utility poles — very thin, tall
        n_poles = int(np.random.randint(*self.cfg["hard_pole_n"]))
        for _ in range(n_poles):
            r = self._r("hard_pole_r")
            h = self._r("hard_pole_h")
            dist = self._r("neg_dist")
            azimuth = np.random.uniform(0, 2 * np.pi)
            actual_dist = pier_envelope_radius + dist + r
            cx = actual_dist * np.cos(azimuth)
            cy = actual_dist * np.sin(azimuth)
            base_z = np.random.uniform(-0.5, 0.5)
            cyl = self._sample_cylinder(r, h, self.cfg["n_hard_neg"], base_z=base_z)
            cyl[:, 0] += cx
            cyl[:, 1] += cy
            cyl = self._add_noise(cyl, self.cfg["noise"])
            pieces.append(cyl)

        # 3. Building columns — rectangular prisms (hardest negatives)
        n_cols = int(np.random.randint(*self.cfg["hard_column_n"]))
        for _ in range(n_cols):
            w = self._r("hard_column_w")
            d = self._r("hard_column_d")
            h = self._r("hard_column_h")
            dist = self._r("neg_dist")
            azimuth = np.random.uniform(0, 2 * np.pi)
            actual_dist = pier_envelope_radius + dist + max(w, d)
            cx = actual_dist * np.cos(azimuth)
            cy = actual_dist * np.sin(azimuth)
            base_z = np.random.uniform(-0.5, 0.5)
            col = self._sample_box(w, d, h, self.cfg["n_hard_neg"], base_z=base_z)
            col[:, 0] += cx
            col[:, 1] += cy
            col = self._add_noise(col, self.cfg["noise"])
            pieces.append(col)

        # 4. Diagonal braces — tilted cylinders
        n_braces = int(np.random.randint(*self.cfg["hard_brace_n"]))
        for _ in range(n_braces):
            r = self._r("hard_brace_r")
            length = self._r("hard_brace_len")
            angle = np.deg2rad(self._r("hard_brace_angle"))
            dist = self._r("neg_dist")
            azimuth = np.random.uniform(0, 2 * np.pi)
            actual_dist = pier_envelope_radius + dist + r
            cx = actual_dist * np.cos(azimuth)
            cy = actual_dist * np.sin(azimuth)
            base_z = np.random.uniform(0, pier_top_z * 0.5)
            start = np.array([cx, cy, base_z])
            direction = np.array([np.cos(azimuth) * np.sin(angle),
                                  np.sin(azimuth) * np.sin(angle),
                                  np.cos(angle)])
            direction = direction / np.linalg.norm(direction)
            brace = self._sample_tilted_cylinder(start, direction, r, length,
                                                  self.cfg["n_hard_neg"])
            brace = self._add_noise(brace, self.cfg["noise"])
            pieces.append(brace)

        if len(pieces) == 0:
            return np.empty((0, 3)), {"type": "hard_negatives", "instances": 0}

        pts = np.vstack(pieces)
        meta = {"type": "hard_negatives",
                "n_trees": n_trees, "n_poles": n_poles,
                "n_columns": n_cols, "n_braces": n_braces}
        return pts, meta

    # ------------------------------------------------------------------
    #  main orchestrator
    # ------------------------------------------------------------------

    def generate_scene(self):
        """Randomly select a pier morphology and assemble the full scene.

        Returns
        -------
        points : (N, 3)  np.float32
        labels : (N, 1)  np.int8      (1 = pier, 0 = context / negatives)
        meta   : dict                  Component-level metadata for debugging.
        """
        pier_type = np.random.choice(["cylinder", "gravity", "y_pier"])

        if pier_type == "cylinder":
            pier_pts, pier_meta = self.generate_cylinder_pier()
        elif pier_type == "gravity":
            pier_pts, pier_meta = self.generate_gravity_pier()
        else:
            pier_pts, pier_meta = self.generate_y_pier()

        deck_pts, deck_meta = self.generate_deck(
            pier_meta["top_z"], pier_meta["pier_width"])
        gnd_pts, gnd_meta = self.generate_ground(pier_base_z=0.0)
        neg_pts, neg_meta = self.generate_negatives(
            pier_meta["pier_width"] / 2, pier_meta["top_z"])
        hard_neg_pts, hard_neg_meta = self.generate_hard_negatives(
            pier_meta["pier_width"] / 2, pier_meta["top_z"])

        # assemble
        all_pts = [pier_pts, deck_pts, gnd_pts]
        all_labels = [np.ones((len(pier_pts), 1), dtype=np.int8),
                      np.zeros((len(deck_pts), 1), dtype=np.int8),
                      np.zeros((len(gnd_pts),  1), dtype=np.int8)]

        if len(neg_pts) > 0:
            all_pts.append(neg_pts)
            all_labels.append(np.zeros((len(neg_pts), 1), dtype=np.int8))

        if len(hard_neg_pts) > 0:
            all_pts.append(hard_neg_pts)
            all_labels.append(np.zeros((len(hard_neg_pts), 1), dtype=np.int8))

        points = np.vstack(all_pts).astype(np.float32)
        labels = np.vstack(all_labels)

        meta = {"pier": pier_meta, "deck": deck_meta,
                "ground": gnd_meta, "negatives": neg_meta,
                "hard_negatives": hard_neg_meta}

        points, labels = self.degrader.degrade(points, labels)
        meta["degradation"] = self.degrader.cfg

        return points, labels, meta
