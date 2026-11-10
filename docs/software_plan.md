# Remaining software plan

Written 2026-10-19. Everything through Module 5 and the simulator is built and
tested (19 suites green). This covers what is left.

---

## Why dynamic obstacles come first

Three experiments have now shown that the safety filter is **not the binding
constraint** in a static world:

| obstacle source | why the CBF didn't matter | measured |
|---|---|---|
| live scan | pose cancels out of `h` — the robot measures its own clearance | B vs C identical, 167.2 vs 167.3 mm |
| SLAM-built map | map and pose drift together, so local avoidance stays correct | 0 violations at 30% slip, 0.8 m pose error |
| prior map | A\* routes through the gap centre; the CBF barely engages | +150 mm radius bought +10 mm clearance |

Something else was already protecting the robot every time. **The CBF is only
load-bearing for an obstacle the planner does not know about.** So the moving
obstacle is not a feature addition — it is the condition that makes the
research question answerable.

---

## What the CBF is for, and when adaptive matters (noted 2026-11-09)

**The CBF and the adaptive margin are two separate pieces.** The CBF is the
safety filter: every control cycle it checks the command pure pursuit wants to
send and, if that command would bring the robot too close to something, changes
it as little as possible. The margin only decides *how close is too close*.
"Make the buffer big enough" still uses the CBF, it just picks a fixed number.
So "adaptive vs fixed" is a question about sizing the buffer, not about whether
to have a CBF.

**The CBF is the seatbelt.** A\* only knows the map and replans every 0.5 s, and
pure pursuit can cut corners or overshoot. The CBF catches what those miss, at
the last moment, and does nothing otherwise. It earns its place when the world
surprises the planner: a person stepping out, a chair not in the map, the robot
drifting off its path. The static simulations had no surprises, which is part of
why every comparison came out flat.

**A bigger buffer is not free.** It makes the robot more cautious everywhere. At
`R = 0.45` the robot could not fit a doorway it should have. On this robot a
buffer big enough to be safe (~0.35) happened to still be cheap, so adaptive had
nothing to add. Adaptive only pays off when the safe buffer would be expensive to
carry all the time.

**Adaptive needs three things at once:** large worst-case uncertainty (big enough
that covering it permanently blocks real paths), uncertainty that is usually
small (so the worst case is rare), and the CBF actually doing the protecting.
This robot fails the first and third: forward sigma is flat at ~10 mm (encoders
pin it) and lateral sigma grows slowly to ~120 mm, and A\* keeps it clear of
mapped walls.

**Wheel slip is not in sigma.** The EKF covariance only models random,
zero-mean error. Slip is a consistent one-direction error the filter cannot see,
so under slip the robot is further off than sigma says and the margin is too
small (15% slip: 822 mm real error vs ~5 mm claimed). For the fall writeup this
is stated as an assumption (error is random and the covariance is honest);
covering it would need a worst-case bound or a slip/scale state in the filter.

**This matches Cosner's own results.** MR-CBF (IROS 2021) used a constant
0.4 m buffer and compared only against no buffer. ORIO (ICRA 2024) compared a
fixed buffer against an adaptive one: adaptive won when uncertainty swung a lot
(~50x near the ground in sim, a swinging slung load on hardware), and did no
better than the simple version in the low-variance drop test. This robot's pose
uncertainty is the drop-test case. Both of his methods also handle consistent
bias, not just spread, which `k·σ` does not.

---

## Phase 1 — Dynamic obstacles

Build order chosen so each step is verifiable before the next introduces error.

### 1.1 Simulator: moving obstacles — **DONE**

`sim/sensors.py`: `raycast_circles`, `SimSensors(movers=[...])`,
`mover_state()` returning ground-truth centre / velocity / radius, and
`true_clearance` accounting for movers.

Verified: ray–circle by hand; a crossing obstacle appears in the scan at the
right moment; clearance tracks it.

### 1.2 CBF: the obstacle-velocity term

`lidar_nav/control/cbf_baseline.py` — `fixed_radius_cbf_constraint(..., v_obs=None)`

```
ḣ  =  a_v·v  +  a_w·ω  −  ê·v_obs
```

`v_obs` is not something the controls can influence, so it moves the **bound**,
not the `G` row:

```
g_row @ u  ≤  γh + ê·v_obs
```

Test with **ground-truth** velocity from `mover_state()`, so the control change
is verified before any estimation error exists.

**Exit criteria**
- `v_obs = 0` reproduces the current constraint exactly (regression)
- an obstacle closing head-on tightens the bound; receding loosens it
- hand-computed case: obstacle at 1.0 m closing at 0.5 m/s, `γ = 1`, `R = 0.25`
- closed loop: robot stops/swerves for a crossing obstacle it was going to hit

### 1.3 Tracking: estimate `v_obs` from the scan

`lidar_nav/control/tracking.py` — new file

```python
cluster_scan(points, max_gap)        -> list of clusters (centroid, extent)
associate(prev_tracks, clusters)     -> nearest-neighbour with a gate
update_tracks(tracks, dt)            -> constant-velocity filter per track
```

A small constant-velocity Kalman filter per track (state `[x, y, vx, vy]`).
Track quality matters: a track seen once has a meaningless velocity, so carry a
hit count and only feed `v_obs` into the barrier once a track is confirmed.

**Exit criteria**
- estimated velocity within ~20% of ground truth for a 0.5 m/s crosser
- static walls do not produce spurious tracks
- the occupancy grid's stale smear behind a mover does not create a phantom

### 1.4 Grid handling for movers

The log-odds grid assumes a static world. A mover leaves occupied cells behind
it. Either decay the grid, or (simpler and better here) feed **tracked movers
directly to the safety filter from the scan**, bypassing the grid, and let the
grid keep handling static structure.

Decide and record the reason — it is a defensible line either way.

---

## Phase 2 — The decisive experiment

`experiments/archive/iso_safety.py` already exists; rerun it in a scenario with a mover.

**The question:** at equal minimum clearance, does the adaptive margin complete
more tasks than a retuned fixed radius?

- IV: fixed `R` ∈ {0.25 … 0.45} versus adaptive `k` ∈ {0.5 … 2.0}
- Blocks: obstacle speed, crossing geometry, and whether the mover is in the map
- 15–20 seeds, paired
- Metrics: min true clearance, violation rate, task success **against ground
  truth**, total intervention, time to goal
- Figure: Pareto curve in (clearance, success) space

**Both outcomes are publishable.** If adaptive dominates, that is the result. If
the curves coincide, the honest conclusion is that a fixed radius suffices for
this platform — report it, drop the margin, and keep a well-engineered CBF robot.

Decide this before writing anything up.

---

## Phase 3 — Hardware

Nothing here depends on Phase 2's outcome.

### 3.1 Measure, don't guess
Wheelbase, max forward speed, max turn rate, coast distance after a stop
command, real LiDAR range noise. These replace assumptions in `ControlParams`.
The coast distance is what justifies `R = 0.25`.

### 3.2 `PiSensors`
Same `read()` / `send()` interface as `SimSensors`, so `navigate()` is untouched.
- background thread publishing the latest complete LiDAR revolution
- IMU and encoder reads
- `(v, ω) → wheel speeds` — where `wheelbase` finally gets used:
  `v_l = v − ω·W/2`, `v_r = v + ω·W/2`

### 3.3 Real-time loop
- rate limiter; `dt` from the clock, not assumed
- overrun detection and logging
- EKF `predict` at the IMU rate (~200 Hz), control at 50 Hz
- **watchdog: motors stop if the loop dies or overruns badly**

### 3.4 Performance
`integrate_scan` is 52 ms in sim, so ~150–250 ms on a Pi against a 180 ms budget.
Either vectorize it (accumulate cell indices, one `np.add.at`) or map with every
third beam. Everything else has 4+ orders of magnitude of headroom — the QP is
30 µs.

### 3.5 Hardware metrics
`true_clearance` does not exist off-sim. Metrics 1 and 2 need a surveyed course
or motion capture.

---

## Phase 4 — Writeup

Material that already exists and should not be rediscovered later:

**Findings**
- ICP's normals were fitted over 8 beams (28 mm) against 20 mm of range noise,
  fabricating observability that doesn't exist. Fixed at `k = 32`; covariance
  went from 77× overconfident to 4.5×, and the corridor's anisotropy appeared
  (`σ_x/σ_y = 14.6`).
- Scan matching fused as an absolute pose fix made the filter inconsistent
  (NEES 94). Reformulated as velocity + turn rate: NEES 0.05–0.4.
  *Velocity is observable from two scans; position is not.*
- In a SLAM stack the map and pose drift together, so pose uncertainty does not
  cause collisions. It only bites against a prior map or an untracked obstacle.

**Limitations** (each already quantified)
- first-order kinematics: valid while `v·τ ≪ R` (~5 cm against 25 cm at 0.5 m/s)
- no odometry scale state, so consistent wheel slip is unmodellable
- χ² gate catches blunders, not slow bias — needs a sequential test (CUSUM)
- `k = 2` is a per-step chance constraint, not a guarantee
- point-based obstacle sampling; segment barriers left as future work
- no loop closure, so absolute position drift is unbounded

**Positioning** — the method is not novel; the integration and the empirical
characterisation are. Nearest work:
Cosner et al., *Measurement-Robust CBFs* (IROS 2021) — worst-case, bounded error;
Luo, Sun & Kapoor, *Probabilistic Safety Barrier Certificates* (NeurIPS 2020) —
chance-constrained, multi-robot.

---

## Order of work

```
1.2  CBF velocity term            half a day
1.3  tracking                     1-2 days        <- the hard part
1.4  grid/mover decision          an hour
2    the decisive experiment      1 day
3    hardware                     ongoing, parallel
4    writeup                      continuous
```

Phase 2 is the decision point. Do not start Phase 4's results section before it.
