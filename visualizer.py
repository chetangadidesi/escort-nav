import matplotlib.pyplot as plt
import numpy as np

class DashboardVisualizer:
    def __init__(self, sim):
        self.sim = sim
        self.config = sim.env.config
        
        # State Tracking for Plots
        self.dynamic_artists = {
            'vis_fill': [],
            'frontier_lines': [],
            'drone_frontier_lines': [],
            'closest_frontier_lines': []
        }
        
        self.history = {
            'x': [],
            'y_uncovered': [],
            'y_unseen': [],
            'y_area': [],
            'y_threat': [],
            'y_coh': []
        }
        
        self.D_safe = 12.0
        self.D_stop = 6.0

        # Initialize all windows
        self.setup_main_map()
        self.setup_dashboards()

    def setup_main_map(self):
        """Initializes Window 1: The Main Strategic Map."""
        self.fig_map, self.ax_map = plt.subplots(figsize=(10, 10))
        
        # Draw Boundary
        x1, y1 = zip(*self.config["outer_boundary"])
        self.ax_map.plot(x1 + (x1[0],), y1 + (y1[0],), 'k-', linewidth=1, label='Outer Boundary')

        # Draw Obstacles
        for i, hole in enumerate(self.config["holes"]):
            x2, y2 = zip(*hole)
            self.ax_map.plot(x2 + (x2[0],), y2 + (y2[0],), 'k-', linewidth=1)
            self.ax_map.fill(x2, y2, color='gray', alpha=0.5, label='Obstacles' if i == 0 else '')

        # Draw Planned Path
        px, py = zip(*self.sim.frames)
        self.ax_map.plot(px, py, color='steelblue', linestyle='--', linewidth=1, alpha=0.4, label='Planned Path')
        
        # Start & Goal
        self.ax_map.scatter(*self.config["start_pos"], color='blue',  s=50, marker='o', zorder=6, label='Start')
        self.ax_map.scatter(*self.config["goal_pos"],  color='green', s=50, marker='*', zorder=6, label='Goal')

        self.ax_map.set_aspect('equal')
        self.ax_map.set_title(f"Escort Navigation: {self.config.get('name', 'Swarm')}", fontsize=12, fontweight='bold')
        self.ax_map.grid(True, linestyle='--', alpha=0.4)

        # Initialize moving entities
        self.escort_dot, = self.ax_map.plot([], [], 'o', color='crimson', markersize=7, zorder=7, label='Escort')
        self.drone_scatter = self.ax_map.scatter([], [], color='magenta', marker='^', s=60, zorder=8, label='Swarm Drones')
        
        # Legend placeholders for dynamic geometry
        self.ax_map.fill([], [], color='yellow', alpha=0.3, label='Visibility Area')
        self.ax_map.plot([], [], color='red', linestyle='-', linewidth=1.5, label='Frontier')
        self.ax_map.legend(loc='lower right', fontsize=8)
        self.fig_map.tight_layout()

    def setup_dashboards(self):
        """Initializes Windows 2 through 5."""
        # ---------------------------------------------------------
        # WINDOW 2: RISK ANALYSIS (Uncovered Frontiers)
        # ---------------------------------------------------------
        self.fig_stats, self.ax_stats = plt.subplots(figsize=(8, 4))
        self.line_stats, = self.ax_stats.plot([], [], color='firebrick', lw=2, label="Dist to Closest Uncovered Global Frontier")
        
        self.ax_stats.axhline(y=self.D_safe, color='seagreen', linestyle='--', alpha=0.5, label="Safety Threshold")
        self.ax_stats.fill_between([0, 5000], 0, self.D_safe, color='red', alpha=0.05)

        self.ax_stats.set_xlim(0, 100)
        self.ax_stats.set_ylim(0, 40)
        self.ax_stats.set_xlabel("Distance Traveled (m)")
        self.ax_stats.set_ylabel("Distance to Frontier (m)")
        self.ax_stats.grid(True, linestyle=':', alpha=0.5)
        self.ax_stats.legend(loc='upper right', fontsize=8)
        self.fig_stats.tight_layout()

        # ---------------------------------------------------------
        # WINDOW 3: VISUAL MEMORY (Unseen Frontiers)
        # ---------------------------------------------------------
        self.fig_vis, self.ax_vis = plt.subplots(figsize=(8, 4))
        self.line_vis, = self.ax_vis.plot([], [], color='darkred', lw=2, label="Dist to Closest NEVER-SEEN Frontier")
        
        self.ax_vis.axhline(y=self.D_safe, color='seagreen', linestyle='--', alpha=0.5, label="Braking Threshold")
        self.ax_vis.fill_between([0, 5000], 0, self.D_safe, color='red', alpha=0.05)

        self.ax_vis.set_xlim(0, 100)
        self.ax_vis.set_ylim(0, 40)
        self.ax_vis.set_xlabel("Distance Traveled (m)")
        self.ax_vis.set_ylabel("Distance to Unseen Frontier (m)")
        self.ax_vis.grid(True, linestyle=':', alpha=0.5)
        self.ax_vis.legend(loc='upper right', fontsize=8)
        self.fig_vis.tight_layout()

        # ---------------------------------------------------------
        # WINDOW 4: TOTAL VISIBILITY AREA
        # ---------------------------------------------------------
        self.fig_area, self.ax_area = plt.subplots(figsize=(8, 4))
        self.line_area, = self.ax_area.plot([], [], color='purple', lw=2, label="Combined Visibility Area")

        self.ax_area.set_xlim(0, 100)
        self.ax_area.set_ylim(0, 500)
        self.ax_area.set_xlabel("Distance Traveled (m)")
        self.ax_area.set_ylabel("Total Visibility Area (m²)")
        self.ax_area.grid(True, linestyle=':', alpha=0.5)
        self.ax_area.legend(loc='upper right', fontsize=8)
        self.fig_area.tight_layout()

        # ---------------------------------------------------------
        # WINDOW 5: TACTICAL SCORECARD
        # ---------------------------------------------------------
        self.fig_tac, self.ax_threat = plt.subplots(figsize=(8, 4))
        self.ax_coh = self.ax_threat.twinx()
        
        self.line_threat, = self.ax_threat.plot([], [], color='darkorange', lw=2, label="Cumulative Threat Density")
        self.line_coh, = self.ax_coh.plot([], [], color='teal', lw=2, linestyle=':', label="Avg Swarm Dispersion")

        self.ax_threat.set_xlim(0, 100)
        self.ax_threat.set_ylim(0, 10)
        self.ax_coh.set_ylim(0, 20)
        
        self.ax_threat.set_xlabel("Distance Traveled (m)")
        self.ax_threat.set_ylabel("Total Threat Score", color='darkorange')
        self.ax_coh.set_ylabel("Avg Distance from VIP (m)", color='teal')
        self.ax_threat.grid(True, linestyle=':', alpha=0.5)
        
        lines_1, labels_1 = self.ax_threat.get_legend_handles_labels()
        lines_2, labels_2 = self.ax_coh.get_legend_handles_labels()
        self.ax_threat.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=8)
        self.fig_tac.tight_layout()

    def render(self):
        """Reads data from the simulation state and redraws everything."""
        state = self.sim.state
        dist = state['curr_dist']
        self.history['x'].append(dist)

        # =========================================================
        # 1. UPDATE MAP ENTITIES
        # =========================================================
        self.escort_dot.set_data([state['escort_pos'][0]], [state['escort_pos'][1]])
        
        current_coords = [d.pos for d in self.sim.swarm]
        if current_coords:
            self.drone_scatter.set_offsets(current_coords)

        # =========================================================
        # 2. RENDER VISIBILITY POLYGONS
        # =========================================================
        for patch in self.dynamic_artists['vis_fill']: patch.remove()
        self.dynamic_artists['vis_fill'] = []

        # Draw Swarm Visibility (Cyan)
        pure_swarm_vis = state['raw_swarm_vis'].difference(state['escort_vis'])
        if not pure_swarm_vis.is_empty:
            geoms = [pure_swarm_vis] if pure_swarm_vis.geom_type == 'Polygon' else pure_swarm_vis.geoms
            for geom in geoms:
                if hasattr(geom, 'exterior'):
                    x_vis, y_vis = geom.exterior.xy
                    patch = self.ax_map.fill(x_vis, y_vis, color='cyan', alpha=0.3, zorder=1)[0]
                    self.dynamic_artists['vis_fill'].append(patch)

        # Draw Escort Visibility (Yellow)
        if not state['escort_vis'].is_empty:
            geoms = [state['escort_vis']] if state['escort_vis'].geom_type == 'Polygon' else state['escort_vis'].geoms
            for geom in geoms:
                if hasattr(geom, 'exterior'):
                    x_vis, y_vis = geom.exterior.xy
                    patch = self.ax_map.fill(x_vis, y_vis, color='yellow', alpha=0.3, zorder=2)[0]
                    self.dynamic_artists['vis_fill'].append(patch)

        # =========================================================
        # 3. RENDER ALL FRONTIER LINES
        # =========================================================
        for line in self.dynamic_artists['drone_frontier_lines']: line.remove()
        for line in self.dynamic_artists['closest_frontier_lines']: line.remove()
        for line in self.dynamic_artists['frontier_lines']: line.remove()
        self.dynamic_artists['drone_frontier_lines'] = []
        self.dynamic_artists['closest_frontier_lines'] = []
        self.dynamic_artists['frontier_lines'] = []

        # Blue: Raw Swarm Frontiers
        for raw_f in state['drone_frontiers']:
            if raw_f.length < 1.5: continue
            segs = raw_f.geoms if hasattr(raw_f, 'geoms') else [raw_f]
            for seg in segs:
                if hasattr(seg, 'xy'):
                    x_b, y_b = seg.xy
                    drawn, = self.ax_map.plot(x_b, y_b, color='dodgerblue', linestyle='-', linewidth=2.0, zorder=10)
                    self.dynamic_artists['drone_frontier_lines'].append(drawn)

        # Green: Closest Uncovered Frontier
        if state['closest_uncovered_frontier'] is not None:
            raw_f = state['closest_uncovered_frontier']
            segs = raw_f.geoms if hasattr(raw_f, 'geoms') else [raw_f]
            for seg in segs:
                if hasattr(seg, 'xy'):
                    x_g, y_g = seg.xy
                    drawn, = self.ax_map.plot(x_g, y_g, color='lime', linestyle='-', linewidth=5.0, zorder=15)
                    self.dynamic_artists['closest_frontier_lines'].append(drawn)

        # Dark Red: Unseen VIP Frontiers
        for unseen_f in state['unseen_frontiers']:
            segs = unseen_f.geoms if hasattr(unseen_f, 'geoms') else [unseen_f]
            for seg in segs:
                if seg.length > 0.5:
                    seg = seg.simplify(0.1)
                    x_u, y_u = seg.xy
                    drawn, = self.ax_map.plot(x_u, y_u, color='darkred', linestyle='-', linewidth=3.0, alpha=1.0, zorder=10)
                    self.dynamic_artists['frontier_lines'].append(drawn)

        # Faded Red: Seen VIP Frontiers
        for seen_f in state['seen_frontiers']:
            segs = seen_f.geoms if hasattr(seen_f, 'geoms') else [seen_f]
            for seg in segs:
                if seg.length > 0.5:
                    x_s, y_s = seg.xy
                    drawn, = self.ax_map.plot(x_s, y_s, color='red', linestyle='-', linewidth=1.5, alpha=0.3, zorder=9)
                    self.dynamic_artists['frontier_lines'].append(drawn)

        self.fig_map.canvas.draw_idle()

        # =========================================================
        # 4. UPDATE DASHBOARDS
        # =========================================================
        # Dashboard 2: Uncovered Frontiers
        D_min_unc = state['d_min_uncovered']
        self.history['y_uncovered'].append(D_min_unc if D_min_unc != float('inf') else 0)
        self.line_stats.set_data(self.history['x'], self.history['y_uncovered'])
        
        if dist > self.ax_stats.get_xlim()[1] * 0.8:
            self.ax_stats.set_xlim(0, dist + 50)
        if D_min_unc != float('inf') and D_min_unc > self.ax_stats.get_ylim()[1] * 0.8:
            self.ax_stats.set_ylim(0, D_min_unc * 1.3)
            
        status = f"EXPOSED — {D_min_unc:.1f}m" if D_min_unc < self.D_safe else f"SAFE — {D_min_unc:.1f}m"
        color = 'firebrick' if D_min_unc < self.D_safe else 'black'
        self.ax_stats.set_title(f"RISK STATUS: [ {status} ]", color=color, loc='right', fontweight='bold')
        self.fig_stats.canvas.draw_idle()

        # Dashboard 3: Unseen Frontiers
        D_min_unseen = state['d_min_unseen']
        SAFE_CEILING = max(100.0, self.ax_vis.get_ylim()[1])
        self.history['y_unseen'].append(D_min_unseen if D_min_unseen != float('inf') else SAFE_CEILING)
        self.line_vis.set_data(self.history['x'], self.history['y_unseen'])

        if dist > self.ax_vis.get_xlim()[1] * 0.8:
            self.ax_vis.set_xlim(0, dist + 50)
        if D_min_unseen != float('inf') and D_min_unseen > self.ax_vis.get_ylim()[1] * 0.8:
            self.ax_vis.set_ylim(0, D_min_unseen * 1.3)
            
        status = f"EXPOSED — {D_min_unseen:.1f}m" if D_min_unseen < self.D_safe else f"CLEARED — {D_min_unseen:.1f}m"
        color = 'darkred' if D_min_unseen < self.D_safe else 'black'
        self.ax_vis.set_title(f"TRUE UNKNOWN RISK: [ {status} ]", color=color, loc='right', fontweight='bold')
        self.fig_vis.canvas.draw_idle()

        # Dashboard 4: Area
        area = state['current_area']
        self.history['y_area'].append(area)
        self.line_area.set_data(self.history['x'], self.history['y_area'])
        
        if dist > self.ax_area.get_xlim()[1] * 0.8:
            self.ax_area.set_xlim(0, dist + 50)
        if area > self.ax_area.get_ylim()[1] * 0.9:
            self.ax_area.set_ylim(0, area * 1.2)
            
        self.ax_area.set_title(f"Visibility Area: [ {area:.1f} m² ]", fontweight='bold')
        self.fig_area.canvas.draw_idle()

        # Dashboard 5: Tactical Metrics
        threat = state['total_threat']
        coh = state['avg_dispersion']
        
        self.history['y_threat'].append(threat)
        self.history['y_coh'].append(coh)
        
        self.line_threat.set_data(self.history['x'], self.history['y_threat'])
        self.line_coh.set_data(self.history['x'], self.history['y_coh'])
        
        if dist > self.ax_threat.get_xlim()[1] * 0.8:
            self.ax_threat.set_xlim(0, dist + 50)
        if threat > self.ax_threat.get_ylim()[1] * 0.8:
            self.ax_threat.set_ylim(0, threat * 1.5)
        if coh > self.ax_coh.get_ylim()[1] * 0.8:
            self.ax_coh.set_ylim(0, coh * 1.5)

        breach_text = f"Number of Times drone sees a frontier first: {state['breach_count']}"
        if state['is_breached']:
            self.ax_threat.set_title(f"{breach_text} [ACTIVE ALARM]", color='red', fontweight='bold')
        else:
            self.ax_threat.set_title(breach_text, color='black', fontweight='bold')
            
        self.fig_tac.canvas.draw_idle()