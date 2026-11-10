"""Harder test worlds: a maze, a cluttered room, and a crowd of moving people."""

import numpy as np


def box(cx, cy, size):
    h = size / 2
    x0, x1, y0, y1 = cx - h, cx + h, cy - h, cy + h
    return [[x0, y0, x1, y0], [x1, y0, x1, y1], [x1, y1, x0, y1], [x0, y1, x0, y0]]


def room(x0=1.0, y0=1.0, x1=9.0, y1=9.0):
    return [[x0, y0, x1, y0], [x1, y0, x1, y1], [x1, y1, x0, y1], [x0, y1, x0, y0]]


def maze():
    """The straight line to the goal runs down a corridor that is a dead end.

    The robot treats unexplored space as open, so its first plan goes straight down
    the corridor. With a 3 m sensor range the end wall only appears once it is
    inside, so it has to stop, turn around and take the long way below. The space
    above the corridor is a second pocket, closed off at the corridor mouth.
    """
    seg = room() + [
        [3.0, 5.7, 7.4, 5.7],
        [3.0, 4.3, 7.4, 4.3],
        [7.4, 4.3, 7.4, 5.7],
        [3.0, 5.7, 3.0, 9.0],
        [5.0, 1.0, 5.0, 2.6],
    ]
    return dict(name="maze", segments=np.array(seg, float), start=[1.8, 5.0, 0.0],
                goal=[8.3, 5.0], grid_wh=(10.0, 10.0), z_max=3.0, movers=[], max_cycles=4000,
                dead_end=dict(x0=3.0, x1=7.4, y0=4.3, y1=5.7))


def doorway():
    """A room split by a wall with a single 1.2 m doorway."""
    seg = room() + [[5.0, 1.0, 5.0, 4.0], [5.0, 5.2, 5.0, 9.0]]
    return dict(name="doorway", segments=np.array(seg, float), start=[2.0, 5.0, 0.0],
                goal=[8.0, 5.0], grid_wh=(10.0, 10.0), z_max=12.0, movers=[], max_cycles=2500)


def clutter():
    boxes = [(3.0, 5.0, 0.6), (3.2, 3.2, 0.5), (3.0, 6.9, 0.5), (4.7, 4.1, 0.5),
             (4.6, 6.0, 0.6), (4.8, 2.2, 0.4), (5.0, 7.9, 0.4), (6.2, 5.2, 0.6),
             (6.4, 3.3, 0.5), (6.1, 7.2, 0.5), (7.6, 4.3, 0.4), (7.5, 6.2, 0.4)]
    seg = room()
    for b in boxes:
        seg += box(*b)
    return dict(name="clutter", segments=np.array(seg, float), start=[1.8, 5.0, 0.0],
                goal=[8.3, 5.0], grid_wh=(10.0, 10.0), z_max=12.0, movers=[],
                max_cycles=4000)


def crowd():
    seg = room() + box(5.0, 2.0, 0.5) + box(5.0, 8.0, 0.5)
    movers = [
        dict(start=(3.0, 2.6), vel=(0.0, 0.40), span=4.8, radius=0.22),
        dict(start=(4.2, 7.4), vel=(0.0, -0.30), span=4.8, radius=0.22, phase=1.5),
        dict(start=(5.4, 2.6), vel=(0.0, 0.50), span=4.8, radius=0.22, phase=3.0),
        dict(start=(6.6, 7.4), vel=(0.0, -0.35), span=4.8, radius=0.22, phase=0.8),
        dict(start=(2.6, 6.2), vel=(0.25, 0.0), span=5.0, radius=0.22),
        dict(start=(7.6, 3.8), vel=(-0.20, 0.0), span=4.5, radius=0.22, phase=2.0),
    ]
    return dict(name="crowd", segments=np.array(seg, float), start=[1.8, 5.0, 0.0],
                goal=[8.3, 5.0], grid_wh=(10.0, 10.0), z_max=12.0, movers=movers,
                max_cycles=4000)


WORLDS = {"doorway": doorway, "clutter": clutter, "crowd": crowd, "maze": maze}
