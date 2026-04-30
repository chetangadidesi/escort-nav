import numpy as np
import networkx as nx
from shapely.geometry import LineString
from visibility_kernel import containing_triangle, next_edge

def compute_mesh_path(env, start_pos, goal_pos):
    d = env.d
    start_tri = containing_triangle(d, start_pos[0], start_pos[1])
    goal_tri = containing_triangle(d, goal_pos[0], goal_pos[1])
    
    if start_tri == -1 or goal_tri == -1: 
        return [start_pos, goal_pos]
        
    try:
        tri_path = nx.shortest_path(env.nav_graph, source=start_tri, target=goal_tri, weight='weight')
    except nx.NetworkXNoPath: 
        return [start_pos, goal_pos]
        
    path = [start_pos]
    for i in range(len(tri_path) - 1):
        t1 = tri_path[i]; t2 = tri_path[i+1]
        for j in range(3):
            edg = t1 * 3 + j
            if d['halfedges'][edg] // 3 == t2:
                p1 = d['coords'][d['triangles'][edg]]
                p2 = d['coords'][d['triangles'][next_edge(edg)]]
                path.append(((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0))
                break
    path.append(goal_pos)
    return path

def smoothing_path(env, path):
    if len(path) <= 2: return path
    smoothed_path = [path[0]]
    current_idx = 0
    buffer_margin = 1e-3 
    
    while current_idx < len(path) - 1:
        furthest_visible_idx = current_idx + 1
        for lookahead_idx in range(len(path) - 1, current_idx + 1, -1):
            line = LineString([path[current_idx], path[lookahead_idx]])
            collision = False
            for obs in env.obstacles:
                if line.intersects(obs.buffer(buffer_margin)):
                    collision = True; break
            if not collision and not line.within(env.boundary_poly.buffer(buffer_margin)):
                collision = True
            if not collision:
                furthest_visible_idx = lookahead_idx; break 
        smoothed_path.append(path[furthest_visible_idx])
        current_idx = furthest_visible_idx
    return smoothed_path

def interpolate_path(path, step_size=0.8):
    interpolated = [path[0]]
    for i in range(len(path) - 1):
        p1, p2 = np.array(path[i]), np.array(path[i + 1])
        n_steps = max(1, int(np.linalg.norm(p2 - p1) / step_size))
        for j in range(1, n_steps + 1):
            interpolated.append(tuple(p1 + (j / n_steps) * (p2 - p1)))
    return interpolated