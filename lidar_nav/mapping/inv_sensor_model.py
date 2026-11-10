import math

def classify_beam(z, z_max, r_min, clearing_dist):
    """
    Decide what one beam's range reading means and how far to raycast for it. \n
    returns        (verdict, ray_range) with verdict "valid_hit" or "no_return",
                   or None if the beam carries no usable information, 
    """

    if not math.isfinite(z) or z < r_min:
        return None
    elif z >= z_max:
        return "no_return", clearing_dist
    else:
        return "valid_hit", z