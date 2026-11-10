def bresenham_cells(x0, y0, x1, y1):
    """
    Integer grid traversal from cell (x0,y0) to cell (x1,y1) INCLUSIVE.

    Returns an ordered list of (cx, cy) integer cell coordinates along the
    line, starting AT (x0,y0) and ending AT (x1,y1). 
    """

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    error = dx - dy
    x = x0
    y = y0
    cells = []
    while True:
        cells.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * error
        if e2 > -dy:
            error -= dy
            x += sx
        if e2 < dx:
            error += dx
            y += sy
    return cells
