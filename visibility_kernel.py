from shapely.geometry import Polygon, Point
import numpy as np
import math
# ==============================================================================
# GEOMETRY & VISIBILITY KERNEL
# ==============================================================================
epsilon = 1.1102230246251565e-16
splitter = 134217729.0

def two_sum(a,b):
    x = a + b; b_virtual = x - a; a_virtual = x - b_virtual
    b_roundoff = b - b_virtual; a_roundoff = a - a_virtual
    return x, a_roundoff + b_roundoff

def split(a):
    c = splitter * a; a_big = c - a; a_hi = c - a_big; a_lo = a - a_hi
    return a_hi, a_lo

def two_product(a, b):
    x = a * b; a_hi, a_lo = split(a); b_hi, b_lo = split(b)
    err1 = x - (a_hi * b_hi); err2 = err1 - (a_lo * b_hi); err3 = err2 - (a_hi * b_lo)
    y = a_lo * b_lo - err3
    return x, y

def orient2d_adapt(pa, pb, pc):
    acx = pa[0] - pc[0]; bcx = pb[0] - pc[0]; acy = pa[1] - pc[1]; bcy = pb[1] - pc[1]
    detleft, detleft_err = two_product(acx, bcy); detright, detright_err = two_product(acy, bcx)
    det, det_err = two_sum(detleft, -detright)
    b_virtual = det - detleft; a_virtual = det - b_virtual
    b_roundoff = -detright - b_virtual; a_roundoff = detleft - a_virtual
    det_err += a_roundoff + b_roundoff
    return det_err

def orient2d(pa, pb, pc):
    det = (pa[0] - pc[0]) * (pb[1] - pc[1]) - (pa[1] - pc[1]) * (pb[0] - pc[0])
    if det != 0.0:
        det_bound = (abs(pa[0] - pc[0]) + abs(pb[0] - pc[0])) * (abs(pa[1] - pc[1]) + abs(pb[1] - pc[1]))
        if abs(det) >= epsilon * det_bound: return det
    return orient2d_adapt(pa, pb, pc)

def build_halfedges(triangles_flat):
    num_triangles = len(triangles_flat) // 3
    halfedges = -np.ones(num_triangles * 3, dtype=int)
    edge_map = {}
    for t_idx in range(num_triangles):
        for i in range(3):
            idx1 = triangles_flat[t_idx * 3 + i]
            idx2 = triangles_flat[t_idx * 3 + (i + 1) % 3]
            edge = tuple(sorted((idx1, idx2)))
            if edge not in edge_map: edge_map[edge] = []
            edge_map[edge].append(t_idx * 3 + i)
    for hes in edge_map.values():
        if len(hes) == 2:
            he1, he2 = hes
            halfedges[he1], halfedges[he2] = he2, he1
    return halfedges

def next_edge(e): return e - 2 if e % 3 == 2 else e + 1
def prev_edge(e): return e + 2 if e % 3 == 0 else e - 1
def is_left_of(x1, y1, x2, y2, px, py): return orient2d((x1, y1), (x2, y2), (px, py)) > 0
def is_right_of(x1, y1, x2, y2, px, py): return orient2d((x1, y1), (x2, y2), (px, py)) < 0

def order_angles(qx, qy, p1x, p1y, p2x, p2y):
    seg_left = is_left_of(qx, qy, p2x, p2y, p1x, p1y)
    lx, ly = (p1x, p1y) if seg_left else (p2x, p2y)
    rx, ry = (p2x, p2y) if seg_left else (p1x, p1y)
    return [lx, ly, rx, ry]

def order_del_angles(d, qx, qy, p1_idx, p2_idx):
    coords = d['coords']
    p1x, p1y = coords[p1_idx]; p2x, p2y = coords[p2_idx]
    return order_angles(qx, qy, p1x, p1y, p2x, p2y)

def is_within_cone(px, py, slx, sly, srx, sry, rlx, rly, rrx, rry):
    if is_left_of(px, py, slx, sly, rrx, rry): return False
    if is_left_of(px, py, rlx, rly, srx, sry): return False
    return True

def restrict_angles(px, py, slx, sly, srx, sry, rlx, rly, rrx, rry):
    nlx, nly, res_left = (slx, sly, True) if is_right_of(px, py, rlx, rly, slx, sly) else (rlx, rly, False)
    nrx, nry, res_right = (srx, sry, True) if is_left_of(px, py, rrx, rry, srx, sry) else (rrx, rry, False)
    return ([nlx, nly, nrx, nry], res_left, res_right)

def seg_intersect_ray(s1x, s1y, s2x, s2y, r1x, r1y, r2x, r2y):
    rdx, rdy = r2x - r1x, r2y - r1y
    sdx, sdy = s2x - s1x, s2y - s1y
    denominator = sdx * rdy - sdy * rdx
    if denominator == 0: return float('inf')
    t2 = (rdx * (s1y - r1y) + rdy * (r1x - s1x)) / denominator
    if rdx != 0: t1 = (s1x + sdx * t2 - r1x) / rdx
    else: t1 = (s1y + sdy * t2 - r1y) / rdy if rdy != 0 else float('inf')
    if t1 < -1e-9 or t2 < -1e-9 or t2 > 1.0 + 1e-9: return float('inf')
    return t1

def containing_triangle(d, qx, qy):
    coords = d['coords']
    triangles = d['triangles']
    q = (qx, qy)
    pt = Point(qx, qy)
    for t_idx in d['tri_strtree'].query(pt):
        p_indices = triangles[t_idx*3 : t_idx*3+3]
        p1, p2, p3 = coords[p_indices]
        if orient2d(p1, p2, q) >= 0 and orient2d(p2, p3, q) >= 0 and orient2d(p3, p1, q) >= 0:
            return t_idx
    return -1

def triangular_expansion(d, qx, qy, obstructs):
    memo = {}
    triangles = d['triangles']
    coords = d['coords']
    halfedges = d['halfedges']
    
    def expand(edg_in, rlx, rly, rrx, rry):
        key = (edg_in, rlx, rly, rrx, rry)
        if key in memo: return memo[key]
        ret = []
        edges = [next_edge(edg_in), prev_edge(edg_in)]
        for edg in edges:
            p1_idx, p2_idx = triangles[edg], triangles[next_edge(edg)]
            adj_out = halfedges[edg]
            slx, sly, srx, sry = order_del_angles(d, qx, qy, p1_idx, p2_idx)
            if not is_within_cone(qx, qy, slx, sly, srx, sry, rlx, rly, rrx, rry): continue
            [nlx, nly, nrx, nry], res_l, res_r = restrict_angles(qx, qy, slx, sly, srx, sry, rlx, rly, rrx, rry)
            if orient2d((qx, qy), (nrx, nry), (nlx, nly)) <= 0.0: continue
            if adj_out != -1 and not obstructs(edg):
                ret.extend(expand(adj_out, nlx, nly, nrx, nry))
                continue
            if not res_l:
                inter = seg_intersect_ray(slx, sly, srx, sry, qx, qy, rlx, rly)
                if inter != float('inf'): slx, sly = qx + inter * (rlx-qx), qy + inter * (rly-qy)
            if not res_r:
                inter = seg_intersect_ray(slx, sly, srx, sry, qx, qy, rrx, rry)
                if inter != float('inf'): srx, sry = qx + inter * (rrx-qx), qy + inter * (rry-qy)
            ret.append((slx, sly, srx, sry))
        memo[key] = ret
        return ret

    tri_start = containing_triangle(d, qx, qy)
    if tri_start == -1: return []
    ret = []
    p_indices = triangles[tri_start*3 : tri_start*3+3]
    points = coords[p_indices]
    points_sorted = sorted(points.tolist(), key=lambda p: np.arctan2(p[1] - qy, p[0] - qx))
    for i in range(3):
        p_start, p_end = points_sorted[i], points_sorted[(i + 1) % 3]
        rlx, rly, rrx, rry = order_angles(qx, qy, p_start[0], p_start[1], p_end[0], p_end[1])
        for edg in [tri_start * 3, tri_start * 3 + 1, tri_start * 3 + 2]:
            p1_idx, p2_idx = triangles[edg], triangles[next_edge(edg)]
            p1c, p2c = coords[p1_idx], coords[p2_idx]
            if (np.allclose(p1c, p_start) and np.allclose(p2c, p_end)) or (np.allclose(p1c, p_end) and np.allclose(p2c, p_start)):
                adj = halfedges[edg]
                if adj == -1 or obstructs(edg):
                    ret.append(order_angles(qx, qy, p1c[0], p1c[1], p2c[0], p2c[1]))
                else:
                    ret.extend(expand(adj, rlx, rly, rrx, rry))
                break
    return ret

def get_visibility_polygon(viewpoint, d, obstructs_func):
    segs = triangular_expansion(d, viewpoint[0], viewpoint[1], obstructs_func)
    if not segs: return Polygon()
    qx, qy = viewpoint
    pts = {}
    for s in segs:
        for px, py in [(s[0], s[1]), (s[2], s[3])]:
            angle = math.atan2(py - qy, px - qx)
            pts[angle] = (px, py)
            
    if len(pts) < 3: 
        return Polygon()
        
    ordered = [pts[a] for a in sorted(pts)]
    try:
        poly = Polygon(ordered)
        if poly.area < 1e-6:
            return Polygon()
        return poly
    except ValueError:
        return Polygon()
