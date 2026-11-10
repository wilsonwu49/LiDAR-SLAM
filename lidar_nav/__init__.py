"""The robot code: everything that would run on the real robot.

    estimation/     stage 1, EKF
    scan_matching/  stage 2, ICP
    mapping/        stage 3, occupancy grid
    planning/       stage 4, A*
    control/        stage 5, pure pursuit + CBF safety filter
    navigate.py     the loop that runs all five
"""
