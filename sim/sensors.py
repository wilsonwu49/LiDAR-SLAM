"""Simulated sensors and ground truth, so navigate() can run without hardware.

This module owns the TRUE pose. The EKF never sees it -- it is the reference
F.3 metrics 1 and 2 are measured against.

Same read()/send() interface a hardware PiSensors will expose, so navigate()
does not change when the robot arrives.
"""

import numpy as np

G = 9.80665


def raycast_segments(origin, bearings, segments, z_max, theta=0.0):
    """
    Range along each beam to the nearest wall segment, inf where nothing is hit.

    origin    (2,) sensor position in the world frame
    bearings  (N,) beam angles in the BODY frame [rad]
    segments  (M, 4) walls as [x0, y0, x1, y1] in the world frame
    z_max     maximum range; beyond this a beam returns inf
    theta     robot heading, to turn body bearings into world directions

    returns   (N,) range per beam [m]
    """
    d = np.stack([np.cos(bearings + theta), np.sin(bearings + theta)], axis=1)  # (N,2)
    a = segments[:, :2]                                   # (M,2) segment start
    e = segments[:, 2:] - a                               # (M,2) segment vector

    # ray:  origin + t*d       segment:  a + s*e,  0 <= s <= 1,  t >= 0
    # solve by 2D cross products:  t = (a-o) x e / (d x e),  s = (a-o) x d / (d x e)
    ao = a[None, :, :] - origin[None, None, :]            # (1,M,2) -> broadcast
    dxe = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]      # (N,M)
    aoxe = ao[..., 0] * e[None, :, 1] - ao[..., 1] * e[None, :, 0]           # (N,M)
    aoxd = ao[..., 0] * d[:, None, 1] - ao[..., 1] * d[:, None, 0]           # (N,M)

    with np.errstate(divide="ignore", invalid="ignore"):
        t = aoxe / dxe
        s = aoxd / dxe

    hit = (np.abs(dxe) > 1e-12) & (t > 0) & (s >= 0.0) & (s <= 1.0)
    t = np.where(hit, t, np.inf)
    r = t.min(axis=1)
    return np.where(r > z_max, np.inf, r)


class SimSensors:
    """
    Ground-truth world plus synthetic IMU, encoder and LiDAR.

    Call read() once per control cycle and send(u) once per control cycle.
    """

    def __init__(self, segments, pose_true, params, dt=0.02,
                 scan_hz=5.5, n_beams=1080, z_max=12.0,
                 sigma_range=0.02, sigma_accel=0.05, sigma_gyro=0.005,
                 sigma_encoder=0.01, tau_actuation=0.10, slip_frac=0.0,
                 seed=0):
        """
        segments        (M, 4) walls [x0, y0, x1, y1]
        pose_true       (3,) starting [x, y, theta] -- the TRUTH, not an estimate
        params          ControlParams
        dt              control period [s]
        scan_hz         LiDAR revolution rate; a new scan every round(1/(scan_hz*dt)) cycles
        n_beams         beams per revolution
        z_max           LiDAR max range [m]
        sigma_range     per-beam range noise [m]
        sigma_accel     accelerometer noise [m/s^2]
        sigma_gyro      gyro noise [rad/s]
        sigma_encoder   body-velocity measurement noise [m/s]
        tau_actuation   first-order lag from commanded to actual velocity [s]
        slip_frac       fraction of commanded motion lost to wheel slip, 0 = none
        seed            RNG seed, so a trial is reproducible
        """
        self.segments = np.asarray(segments, dtype=float).reshape(-1, 4)
        self.pose_true = np.array(pose_true, dtype=float)
        self.params = params
        self.dt = dt
        self.bearings = np.linspace(-np.pi, np.pi, n_beams, endpoint=False)
        self.scan_every = max(1, int(round(1.0 / (scan_hz * dt))))
        self.z_max = z_max

        self.sigma_range = sigma_range
        self.sigma_accel = sigma_accel
        self.sigma_gyro = sigma_gyro
        self.sigma_encoder = sigma_encoder
        self.tau = tau_actuation
        self.slip_frac = slip_frac
        self.rng = np.random.default_rng(seed)

        self.u_actual = np.zeros(2)      # what the wheels are really doing
        self.u_cmd = np.zeros(2)         # what was last commanded
        self.v_prev = 0.0                # for the v_dot finite difference
        self.cycle = 0
        self.scan_seq = -1               # increments once per LiDAR revolution
        self._scan_cache = None          # (angles, ranges), reused between revolutions
        self.trail = [self.pose_true.copy()]

    # ------------------------------------------------------------------ sensing

    def _scan(self):
        """A fresh noisy scan from the true pose. Only called once per revolution."""
        r = raycast_segments(self.pose_true[:2], self.bearings, self.segments,
                             self.z_max, self.pose_true[2])
        finite = np.isfinite(r)
        r = r.copy()
        r[finite] += self.rng.normal(0.0, self.sigma_range, finite.sum())
        return self.bearings.copy(), r

    def read(self):
        """
        One cycle of sensor data.

        returns
          imu       (a_m, w_m) -- specific force and angular rate, body frame
          encoder   (3,) measured body-frame velocity
          scan      (angles, ranges); the SAME object for ~1/(scan_hz*dt) cycles
          scan_seq  revolution counter; unchanged means "no new scan this cycle"
        """
        v, w = self.u_actual
        v_dot = (v - self.v_prev) / self.dt
        self.v_prev = v

        # a_body = [v_dot, v*omega, 0]; the v*omega term is centripetal and must
        # be here or the IMU contradicts the motion on every turn. Subtracting
        # R^T g at level attitude puts +G on the z channel, matching the
        # a_true = R(a_m - b) + g convention in propagate_nominal.
        a_m = np.array([v_dot, v * w, G]) + self.rng.normal(0.0, self.sigma_accel, 3)
        w_m = np.array([0.0, 0.0, w]) + self.rng.normal(0.0, self.sigma_gyro, 3)

        # encoder_h_and_H predicts h = R^T v, i.e. body-frame velocity
        encoder = np.array([v, 0.0, 0.0]) + self.rng.normal(0.0, self.sigma_encoder, 3)

        # one revolution lasts scan_every control cycles; handing back a fresh
        # scan every cycle would simulate a better LiDAR than you own
        if self._scan_cache is None or self.cycle % self.scan_every == 0:
            self._scan_cache = self._scan()
            self.scan_seq += 1

        return (a_m, w_m), encoder, self._scan_cache, self.scan_seq

    # ------------------------------------------------------------------- acting

    def send(self, u):
        """
        Apply a command imperfectly and advance the true pose by dt.

        u   (2,) [v, omega] from the controller
        """
        self.u_cmd = np.asarray(u, dtype=float)

        # first-order lag: the wheels chase the command with time constant tau
        alpha = self.dt / (self.tau + self.dt)
        self.u_actual = self.u_actual + alpha * (self.u_cmd - self.u_actual)

        # slip reduces the ACHIEVED motion but not what the encoder reports --
        # that asymmetry is exactly why odometry cannot detect slip, and why the
        # chi-squared gate on the ICP update is the only thing that can
        v_true = self.u_actual[0] * (1.0 - self.slip_frac)
        w_true = self.u_actual[1] * (1.0 - self.slip_frac)

        self.pose_true[0] += v_true * np.cos(self.pose_true[2]) * self.dt
        self.pose_true[1] += v_true * np.sin(self.pose_true[2]) * self.dt
        self.pose_true[2] += w_true * self.dt

        self.cycle += 1
        self.trail.append(self.pose_true.copy())

    # -------------------------------------------------------------- ground truth

    def true_clearance(self, r_physical):
        """
        Distance from the true pose to the nearest wall, minus the robot's radius.

        This is F.3 metric 1. It uses pose_true and the real segments, never the
        EKF -- the entire point of the experiment is that the two differ.
        """
        p = self.pose_true[:2]
        a = self.segments[:, :2]
        e = self.segments[:, 2:] - a
        L2 = np.einsum("ij,ij->i", e, e)
        # projection parameter, clamped so the closest point stays on the segment
        t = np.where(L2 > 1e-12,
                     np.einsum("ij,ij->i", p - a, e) / np.maximum(L2, 1e-12),
                     0.0)
        closest = a + np.clip(t, 0.0, 1.0)[:, None] * e
        return float(np.linalg.norm(p - closest, axis=1).min()) - r_physical
