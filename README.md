# LiDAR SLAM navigation robot

A differential-drive robot that maps a room with a 2D LiDAR, plans a route and
drives it without hitting anything. Fall goal: in simulation, compare a fixed
CBF safety radius against one scaled by the EKF's pose uncertainty.

## How a cycle works (50 Hz, `lidar_nav/navigate.py`)

```
sense -> estimate -> map -> plan -> control -> act
         (EKF+ICP)   (grid)  (A*)   (pure pursuit + CBF)
```

## Folder map

```
lidar_nav/              the robot code: what would run on the real robot
  navigate.py           the main loop that runs every stage
  estimation/           stage 1  error-state EKF: IMU predict, encoder and ICP-rate updates
  scan_matching/        stage 2  point-to-line ICP between consecutive scans
  mapping/              stage 3  log-odds occupancy grid, Bresenham ray tracing
  planning/             stage 4  wall inflation, A* search, path smoothing
  control/              stage 5  pure pursuit and the CBF-QP safety filter
sim/                    fake robot and sensors, test worlds, video dashboard
tests/                  pass/fail checks, one folder per stage, plus run_tests.py
experiments/            Part F: fixed vs uncertainty-scaled CBF
  results/              saved data (.npz)
  archive/              earlier experiment, kept for reference
docs/                   software plan, block diagram, videos/
```

The study guides for each stage live one folder up, in `../Guides/`
(Module 1 = estimation ... Module 5 = control).

## Setup

```
pip install -r requirements.txt
```

## Commands (run from this folder)

```
python tests/run_tests.py                 # all fast tests (~20 s)
python tests/run_tests.py --all           # plus full drives in four worlds (~45 s)
python tests/run_tests.py icp             # only suites matching "icp"
python -m tests.planning.test_astar       # one suite

python -m sim.dashboard --world maze      # six-panel video -> docs/videos/maze.mp4
python -m sim.dashboard --world crowd --live   # live window (space pauses, q quits)
                                          # worlds: doorway, clutter, crowd, maze

python -m experiments.final_run           # Part F, 30 paired seeds -> experiments/results/
python -m experiments.final_plots         # figures -> docs/figures/
```

Without `imageio-ffmpeg` the dashboard saves PNG frames instead of a video.
