import heapq
import math
import itertools


def heuristic(cell, goal, connectivity=8):
    """
    Admissible, consistent estimate of the remaining cost from cell to goal.

    cell, goal    (r, c) integer cell indices
    connectivity  4 for cardinal moves only, 8 to allow diagonals
    returns       float, never greater than the true optimal cost
    """
    # dx, dy as absolute index differences
    # connectivity 4 -> Manhattan:  dx + dy
    # connectivity 8 -> octile:     (dx + dy) + (sqrt(2) - 2) * min(dx, dy)
    dx = abs(cell[0] - goal[0])
    dy = abs(cell[1] - goal[1])
    if connectivity == 4:
        return dx + dy
    elif connectivity == 8:
        return dx + dy + (math.sqrt(2) - 2) * min(dx, dy)
    else:
        raise ValueError("Error in heuristic")


def neighbors(cell, lethal_mask, connectivity=8):
    """
    Yield the legal moves out of a cell as (neighbor_cell, step_cost) pairs.

    cell          (r, c) integer cell indices
    lethal_mask   bool array (H, W), True = impassable
    connectivity  4 or 8
    returns       iterable of ((r, c), float) for each legal move
    """
    # 1. cardinal offsets, step cost 1
    # 2. if connectivity == 8, add diagonal offsets, step cost sqrt(2)
    # 3. skip any neighbour outside the array or marked lethal
    # 4. corner rule (8 only): a diagonal is legal ONLY if both orthogonally
    #    adjacent shoulder cells are non-lethal -- otherwise the robot would
    #    clip a corner it cannot physically pass through
    if not (connectivity == 8 or connectivity == 4):
        raise ValueError("Connectivity Issues")

    x, y = cell
    H, W = lethal_mask.shape
    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
    if connectivity == 8:
        d = math.sqrt(2)
        moves += [(-1, -1, d), (-1, 1, d), (1, -1, d), (1, 1, d)]

    cell_list = []
    
    for dx, dy, cost in moves:
        inbounds = 0 <= x + dx < H and 0 <= y + dy < W
        if not inbounds:
            continue 
        lethal = lethal_mask[x + dx, y + dy] == True
        if lethal:
            continue

        if dx != 0 and dy != 0:
            shoulder1 = lethal_mask[x + dx, y] == True
            shoulder2 = lethal_mask[x, y + dy] == True
            if shoulder1 or shoulder2:
                continue
        cell_list.append(((x + dx, y + dy), cost))

    return cell_list


def reconstruct_path(prev, start, goal):
    """Walk came_from backwards from goal to start and return the forward path."""
    path = [goal]
    cell = goal
    while cell != start:
        cell = prev[cell]
        path.append(cell)
    path.reverse()
    return path


def astar(start_cell, goal_cell, lethal_mask, cost_field=None, connectivity=8, max_expansions = 20000):
    """
    Shortest collision-free path over the grid graph.

    start_cell, goal_cell   (r, c) integer cell indices
    lethal_mask             bool array (H, W), True = impassable
    cost_field              optional float array (H, W) of extra per-cell cost
    connectivity            4 or 8, must match the heuristic
    returns                 list of (r, c) from start to goal inclusive, or None
    """
    if lethal_mask[goal_cell]:
        return None

    g_score = {start_cell: 0.0}
    prev = {}
    closed = set()
    counter = itertools.count()

    h0 = heuristic(start_cell, goal_cell, connectivity)
    open_set = [(h0, h0, next(counter), start_cell)]

    expansions = 0
    while open_set:
        f_cost, h_cost, count, cell = heapq.heappop(open_set)

        if cell in closed:
            continue
        if cell == goal_cell:
            return reconstruct_path(prev, start_cell, goal_cell)
        if expansions > max_expansions:
            return None
        closed.add(cell)
        expansions += 1

        for next_cell, step_cost in neighbors(cell, lethal_mask, connectivity):
            tentative = g_score[cell] + step_cost
            if cost_field is not None:
                tentative += cost_field[next_cell]

            if tentative < g_score.get(next_cell, math.inf):
                g_score[next_cell] = tentative
                prev[next_cell] = cell
                h = heuristic(next_cell, goal_cell, connectivity)
                heapq.heappush(open_set, (tentative + h, h, next(counter), next_cell))

    return None