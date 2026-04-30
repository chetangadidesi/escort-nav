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
        self.path_cooldown -= 1
        dist_to_immediate = np.linalg.norm(self.immediate_target - self.pos)
        
        # Instant Brain Reset: If the auction changes my target, wake up!
        if np.linalg.norm(self.target - self.prev_target) > 1.0:
            self.path_cooldown = 0
            self.prev_target = self.target.copy()
        
        # Recalculate A* if cooldown is 0 OR we reached the safe boundary of the corner
        if self.path_cooldown <= 0 or dist_to_immediate < (self.R_safe + 0.5):
            raw_path = compute_mesh_path(env, tuple(self.pos), tuple(self.target))
            waypoints = smoothing_path(env, raw_path)
            
            if len(waypoints) > 1:
                self.immediate_target = np.array(waypoints[1])
            else:
                self.immediate_target = self.target
                
            self.path_cooldown = 10 
            
        error = self.immediate_target - self.pos
        dist = np.linalg.norm(error)
        if dist < 0.1: return np.array([0.0, 0.0])
        
        vel = self.Kp * error
        speed = np.linalg.norm(vel)
        if speed > self.max_speed:
            vel = (vel / speed) * self.max_speed
        return vel

    def update_kinematics(self, dt, env, escort_pos):
        """Applies Euler Integration with Fast STRtree Raycasting."""
        u_hat = self.get_nominal_velocity(env)
        
        # Call the new modular CBF
        u_safe = apply_cbf(self.pos, u_hat, self.R_safe, self.gamma, env)

        proposed_pos = self.pos + u_safe * dt
        movement_line = LineString([self.pos, proposed_pos])

        min_inter_dist = float('inf')

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
            safe_dist = max(0.0, min_inter_dist - 0.2)
            vec = proposed_pos - self.pos
            total_dist = np.linalg.norm(vec)
            if total_dist > 1e-5:
                direction = vec / total_dist
                self.pos = self.pos + direction * safe_dist
        else:
            self.pos = proposed_pos