import numpy as np
from shapely.geometry import Point
from shapely.ops import nearest_points

def apply_cbf(pos, u_hat, R_safe, gamma, env):
    """The O(1) Closed-Form CBF Referee."""
    drone_pt = Point(pos)  # the current position of the drone packaged as a shapely object
    
    p1_b, p2_b = nearest_points(drone_pt, env.boundary_geom) # nearest point gives closest points to each other. In this case we get the point on the boundary which is the closest to the drone_pt
    min_dist = p1_b.distance(p2_b) # this gives us the distance between the two points
    closest_wall_pt = np.array([p2_b.x, p2_b.y]) # this the point on that boundary
    
    if env.obstacles:
        closest_obs_idx = env.tree.nearest(drone_pt)
        closest_obs = env.obstacles[closest_obs_idx]
        p1_o, p2_o = nearest_points(drone_pt, closest_obs)
        dist_obs = p1_o.distance(p2_o)
        
        if dist_obs < min_dist:
            min_dist = dist_obs
            closest_wall_pt = np.array([p2_o.x, p2_o.y])
            
    v_to_drone = pos - closest_wall_pt
    d_obs = np.linalg.norm(v_to_drone)
    n_obs = v_to_drone / d_obs if d_obs > 1e-5 else np.array([1.0, 0.0])
    h_obs = d_obs - R_safe
    
    b = -gamma * h_obs
    A_dot_u = np.dot(n_obs, u_hat)
    
    if A_dot_u >= b:
        return u_hat
    else:
        return u_hat + (b - A_dot_u) * n_obs