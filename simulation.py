import numpy as np
from shapely.ops import unary_union
from shapely.geometry import Polygon, LineString

from environment import SwarmEnvironment
from drone import Drone
from path_planning import compute_mesh_path, smoothing_path, interpolate_path
from algorithm1 import assign_swarm_targets

class EscortSimulation:
    def __init__(self, config):
        self.config = config
        self.env = SwarmEnvironment(config)
        
        # Path setup
        raw_path = compute_mesh_path(self.env, config["start_pos"], config["goal_pos"])
        waypoints = smoothing_path(self.env, raw_path) 
        self.frames = interpolate_path(waypoints, step_size=0.6)
        
        # Swarm setup
        fleet_size = config.get("fleet_size", 10)
        self.swarm = []
        radius = 1.0
        for i in range(fleet_size):
            angle = (2 * np.pi / fleet_size) * i
            sx = config["start_pos"][0] + radius * np.cos(angle)
            sy = config["start_pos"][1] + radius * np.sin(angle)
            self.swarm.append(Drone(drone_id=i, start_pos=(sx, sy)))

        # State tracking (Pure Data)
        self.state = {
            'escort_progress': 0.0,
            'vip_paused': False,
            'memory_map': Polygon(),
            'drone_memory': Polygon(),
            
            # Current frame geometry
            'escort_pos': np.array(self.frames[0]),
            'escort_vis': Polygon(),
            'raw_swarm_vis': Polygon(),
            'current_global_vis': Polygon(),
            'unseen_frontiers': [],  # For drawing dark red lines
            'seen_frontiers': [],    # For drawing faded red lines
            'drone_frontiers': [],   # Blue lines
            'closest_uncovered_frontier': None, # Green line
            
            # Metrics for dashboards
            'curr_dist': 0.0,
            'd_min_uncovered': float('inf'),
            'd_min_unseen': float('inf'),
            'current_area': 0.0,
            'total_threat': 0.0,
            'avg_dispersion': 0.0,
            'breach_count': 0,
            'is_breached': False
        }

    def step(self):
        """Runs one tick of the simulation logic."""
        if not self.state['vip_paused']:
            self.state['escort_progress'] += 1.0

        current_idx = int(self.state['escort_progress'])
        if current_idx >= len(self.frames): 
            current_idx = len(self.frames) - 1
            
        pos = self.frames[current_idx]
        self.state['escort_pos'] = np.array(pos)
        self.state['curr_dist'] = current_idx * 0.6
        
        # 1. VISIBILITY CALCULATIONS
        self.state['escort_vis'] = self.env.get_visibility(pos)
        swarm_vis_polys = [self.env.get_visibility(d.pos) for d in self.swarm]
        self.state['raw_swarm_vis'] = unary_union(swarm_vis_polys) if swarm_vis_polys else Polygon()
        self.state['current_global_vis'] = unary_union([self.state['escort_vis'], self.state['raw_swarm_vis']])
        
        # 2. FRONTIER & METRIC CALCULATIONS
        # ==========================================
        DRONE_COVERAGE_RADIUS = 2.5
        D_min_uncovered = float('inf')
        closest_uncovered_frontier = None

        global_frontiers = self.env.get_frontiers(self.state['current_global_vis'])

        for raw_f in global_frontiers:
            if raw_f.length < 1.5:
                continue

            midpoint = np.array(raw_f.centroid.coords[0])
            
            is_physically_covered = any(
                np.linalg.norm(d.pos - midpoint) <= DRONE_COVERAGE_RADIUS
                for d in self.swarm
            )

            if not is_physically_covered:
                dist_to_escort = np.linalg.norm(midpoint - self.state['escort_pos'])
                if dist_to_escort < D_min_uncovered:
                    D_min_uncovered = dist_to_escort
                    closest_uncovered_frontier = raw_f

        # Save the pure data to the state!
        self.state['d_min_uncovered'] = D_min_uncovered
        self.state['closest_uncovered_frontier'] = closest_uncovered_frontier

        # Also calculate the raw drone frontiers here so the visualizer can draw them
        self.state['drone_frontiers'] = self.env.get_frontiers(self.state['raw_swarm_vis'])
        
        # [Insert the mathematical calculations for frontiers and metrics here...]
        # Example metric update:
        self.state['current_area'] = self.state['current_global_vis'].area
        
        # ==========================================
        # VIP THREAT LOGIC & UNSEEN FRONTIERS
        # ==========================================
        active_frontiers = []
        D_min_unseen = float('inf')
        
        raw_escort_frontiers = self.env.get_frontiers(self.state['escort_vis'])
        buffered_drone_memory = self.state['drone_memory'].buffer(0.5) if not self.state['drone_memory'].is_empty else Polygon()
        
        # Calculate heading for directional threat multiplier
        next_idx = min(current_idx + 5, len(self.frames) - 1)
        escort_heading = np.array(self.frames[next_idx]) - self.state['escort_pos']
        heading_mag = np.linalg.norm(escort_heading)
        if heading_mag > 1e-5:
            escort_heading = escort_heading / heading_mag 

        unseen_frontiers_list = []
        seen_frontiers_list = []

        for raw_f in raw_escort_frontiers:
            if raw_f.length < 1.5: continue
            
            midpoint = np.array([(raw_f.xy[0][0] + raw_f.xy[0][-1])/2.0, (raw_f.xy[1][0] + raw_f.xy[1][-1])/2.0])
            width = raw_f.length
            dist_to_escort = np.linalg.norm(midpoint - self.state['escort_pos'])
            
            threat = width / (dist_to_escort**2 + 1e-6) 
            
            # Directional multiplier
            vec_to_frontier = midpoint - self.state['escort_pos']
            vec_mag = np.linalg.norm(vec_to_frontier)
            if vec_mag > 1e-5 and heading_mag > 1e-5:
                dir_to_frontier = vec_to_frontier / vec_mag
                alignment = np.dot(escort_heading, dir_to_frontier) 
                directional_mult = 1.0 + (alignment * 0.5)
                threat *= directional_mult

            # Split mathematically into "Unseen" and "Seen"
            try:
                unseen_f = raw_f.difference(buffered_drone_memory)
                seen_f = raw_f.intersection(buffered_drone_memory)
            except Exception:
                unseen_f = raw_f
                seen_f = LineString() 
            
            if unseen_f.is_empty:
                threat *= 0.5
            else:
                threat *= 2.0
                if dist_to_escort < D_min_unseen: 
                    D_min_unseen = dist_to_escort
                    
            unseen_frontiers_list.append(unseen_f)
            seen_frontiers_list.append(seen_f)

            active_frontiers.append({'threat': threat, 'midpoint': midpoint})
        
        # Save VIP data to state for the visualizer
        self.state['unseen_frontiers'] = unseen_frontiers_list
        self.state['seen_frontiers'] = seen_frontiers_list
        self.state['d_min_unseen'] = D_min_unseen

        # ==========================================
        # TACTICAL SCORECARD METRICS
        # ==========================================
        self.state['total_threat'] = sum(f['threat'] for f in active_frontiers)
        
        if len(self.swarm) > 0:
            self.state['avg_dispersion'] = np.mean([np.linalg.norm(d.pos - self.state['escort_pos']) for d in self.swarm])
        else:
            self.state['avg_dispersion'] = 0.0

        if D_min_unseen != float('inf'):
            if not self.state['is_breached']:
                self.state['breach_count'] += 1
                self.state['is_breached'] = True
        else:
            self.state['is_breached'] = False
        
        # 3. ASSIGN TARGETS & MOVE DRONES
        assign_swarm_targets(self.swarm, active_frontiers, self.state['escort_pos'])

        dt = 0.1
        for drone in self.swarm:
            drone.update_kinematics(dt, self.env, self.state['escort_pos'])
            
        # 4. UPDATE MEMORY
        self.state['memory_map'] = unary_union([self.state['memory_map'], self.state['current_global_vis']])
        self.state['drone_memory'] = unary_union([self.state['drone_memory'], self.state['raw_swarm_vis']])
        
        if int(self.state['escort_progress']) % 10 == 0:
            self.state['memory_map'] = self.state['memory_map'].simplify(0.2)
            self.state['drone_memory'] = self.state['drone_memory'].buffer(0).simplify(0.2)