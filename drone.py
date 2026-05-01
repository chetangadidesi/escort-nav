import numpy as np
from shapely.geometry import Point, LineString
from cbf_collision import apply_cbf
from path_planning import compute_mesh_path, smoothing_path

class Drone:
    def __init__(self, drone_id, start_pos):
        self.id = drone_id
        self.pos = np.array(start_pos, dtype=float)
        self.target = np.array(start_pos, dtype=float)
        self.prev_target = np.array(start_pos, dtype=float)
        
        self.Kp = 5.0         
        self.max_speed = 25.0 
        
        self.R_safe = 1.0     
        self.gamma = 5.0      
        
        self.immediate_target = np.array(start_pos, dtype=float)
        self.path_cooldown = 0

    def get_nominal_velocity(self, env):
        self.path_cooldown -= 1  # start countdown for A-star path planning buffer
        dist_to_immediate = np.linalg.norm(self.immediate_target - self.pos) # a star planner gives path from current drone pos to frontier target and after smoothing each turn is an immediate target
        
        # Instant Brain Reset: If the auction changes my target, wake up!
        if np.linalg.norm(self.target - self.prev_target) > 1.0: # if the hungarian assignment changes the frontier target, then we compare the new one to the one we were going to
            self.path_cooldown = 0 # cool down to 0 so we do path planning again to target
            self.prev_target = self.target.copy()  # copy the target to prev target to do the checks if hungarian changes the target
        
        # Recalculate A* if cooldown is 0 OR we reached the safe boundary of the corner
        if self.path_cooldown <= 0 or dist_to_immediate < (self.R_safe + 0.5):
            raw_path = compute_mesh_path(env, tuple(self.pos), tuple(self.target))
            waypoints = smoothing_path(env, raw_path) # these waypoints are what become the immediate targets
            
            if len(waypoints) > 1:
                self.immediate_target = np.array(waypoints[1]) # assigning the closest waypoint as the immediate target
            else:
                self.immediate_target = self.target # if no waypointst then the final target and the immediate target are the same
                
            self.path_cooldown = 10  # cool down timer reset after finding the waypoints and immediate target
            
        error = self.immediate_target - self.pos  # finding the position error to drive the controller
        dist = np.linalg.norm(error) # norm gives the magnitude of the vector 
        if dist < 0.1: return np.array([0.0, 0.0]) # if we are clsoe to the immediate target then the velocity we need to move with is zero and we replan to the next immediate target
        
        vel = self.Kp * error # finds velocity of drone towards target. simple integrator
        speed = np.linalg.norm(vel) # magnitude
        if speed > self.max_speed: #speed limit check
            vel = (vel / speed) * self.max_speed # setting it to the max allowable speed in the velocity direction
        return vel

    def update_kinematics(self, dt, env, escort_pos):
        """Applies Euler Integration with Fast STRtree Raycasting."""
        u_hat = self.get_nominal_velocity(env)  # find u_hat
        
        # Call the new modular CBF
        u_safe = apply_cbf(self.pos, u_hat, self.R_safe, self.gamma, env) # passing into the cbf to get safe velocity

        proposed_pos = self.pos + u_safe * dt # dt is the time for each frame, we check the final postion of the drone after each frame because this is a discrete simulation
        movement_line = LineString([self.pos, proposed_pos]) # this is the extra distance that is traveled by the drone in the time interval of the frame 

        min_inter_dist = float('inf') # set the intersection distance to inf

        # checks intersections basically a predictive method to not let the drones crash during dt
        def check_intersection(wall_geom):
            nonlocal min_inter_dist
            if movement_line.intersects(wall_geom):
                inter = movement_line.intersection(wall_geom)
                pts = []
                if inter.geom_type == 'Point':
                    pts.append(inter)
                elif inter.geom_type == 'MultiPoint':
                    pts.extend(list(inter.geoms))
                elif inter.geom_type == 'LineString':
                    pts.append(Point(inter.coords[0])) 
                    
                for pt in pts:
                    dist = Point(self.pos).distance(pt)
                    if dist < min_inter_dist:
                        min_inter_dist = dist

        check_intersection(env.boundary_geom)
        possible_matches_idx = env.tree.query(movement_line)
        for idx in possible_matches_idx:
            check_intersection(env.obstacles[idx].boundary)

        if min_inter_dist < float('inf'):
            safe_dist = max(0.0, min_inter_dist - 0.2) # we stop the drone 0.2 meters from the boundary that it hit
            vec = proposed_pos - self.pos 
            total_dist = np.linalg.norm(vec) # finding the magnitude of the vevtor
            if total_dist > 1e-5: 
                direction = vec / total_dist # finding unit vector
                self.pos = self.pos + direction * safe_dist # position is updated to account for the safe distance
        else:
            self.pos = proposed_pos # if no intersections with boundary