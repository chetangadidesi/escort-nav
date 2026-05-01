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
        closest_obs_idx = env.tree.nearest(drone_pt) # finds the nearest obstacle index to the drone
        closest_obs = env.obstacles[closest_obs_idx] # finds the closest obstacle using the index
        p1_o, p2_o = nearest_points(drone_pt, closest_obs) # finds the nearest point on the obstacle from the drone_pt
        dist_obs = p1_o.distance(p2_o) # finds the closest distance from drone to obstacle
        
        if dist_obs < min_dist: # check if the obstacle is closer than the boundary
            min_dist = dist_obs # then the min_distance becomes that distance
            closest_wall_pt = np.array([p2_o.x, p2_o.y]) # now technically the closest wall point becomes the obstacle point
            
    v_to_drone = pos - closest_wall_pt # vector to drone = destination (drone) - origin (closest_wall_pt)
    d_obs = np.linalg.norm(v_to_drone) # length of this vector 
    n_obs = v_to_drone / d_obs if d_obs > 1e-5 else np.array([1.0, 0.0]) # unit vector in that direction
    h_obs = d_obs - R_safe # actual distance including the safety buffer
    
    b = -gamma * h_obs # maximum allowable speed as a function of distance to the boundary
    A_dot_u = np.dot(n_obs, u_hat) # calculates the magnitude of u_hat in the direction of n_obs
    
    if A_dot_u >= b: 
        return u_hat
    else: # pushes the drone outward with a vector with mangnitude of the deficit 
        return u_hat + (b - A_dot_u) * n_obs