import numpy as np
import math
from scipy.optimize import linear_sum_assignment

def assign_swarm_targets(swarm, active_frontiers, escort_pos):
    """Executes target clustering and auction-based assignment for the swarm."""
    active_frontiers.sort(key=lambda x: x['threat'], reverse=True)
    
    top_targets = []
    for f in active_frontiers:
        target_pt = f['midpoint']
        too_close = False
        for selected in top_targets:
            if np.linalg.norm(target_pt - selected) < 5.0: 
                too_close = True
                break
        if not too_close:
            top_targets.append(target_pt)
        if len(top_targets) == len(swarm):
            break 
            
    if top_targets:
        cost_matrix = np.zeros((len(swarm), len(top_targets)))
        for i, drone in enumerate(swarm):
            for j, target in enumerate(top_targets):
                cost = np.linalg.norm(drone.pos - target)
                if np.linalg.norm(drone.target - target) < 2.0:
                    cost -= 50.0  # Persistence hysteresis
                cost_matrix[i, j] = cost
                
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_drones = set()
        for i, j in zip(row_ind, col_ind):
            swarm[i].target = top_targets[j]
            assigned_drones.add(i)
            
        unassigned = set(range(len(swarm))) - assigned_drones
        for i in unassigned:
            hover_offset = np.array([math.cos(i), math.sin(i)]) * 3.0
            swarm[i].target = escort_pos + hover_offset
    else:
        for i in range(len(swarm)):
            hover_offset = np.array([math.cos(i), math.sin(i)]) * 3.0
            swarm[i].target = escort_pos + hover_offset