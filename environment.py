import numpy as np
import math
import networkx as nx
import triangle as tr
from shapely.geometry import Polygon, Point, LineString, box
from shapely.strtree import STRtree
from visibility_kernel import build_halfedges, get_visibility_polygon

class SwarmEnvironment:
    def __init__(self, config):
        self.config = config
        self.boundary_poly = Polygon(config["outer_boundary"])
        self.boundary_geom = self.boundary_poly.boundary
        
        self.obstacles = []
        for hole_coords in config["holes"]:
            self.obstacles.append(Polygon(hole_coords))
            
        self.tree = STRtree(self.obstacles) if self.obstacles else STRtree([Polygon()])
        self.d = self._build_mesh()
        
    def _build_mesh(self):
        vertices = []
        segments = []
        holes_pts = []
        
        def add_ring(coords, is_hole=False):
            if np.allclose(coords[0], coords[-1]): 
                coords = coords[:-1]
            start_idx = len(vertices)
            for x, y in coords: 
                vertices.append([float(x), float(y)])
            for i in range(len(coords)): 
                segments.append([start_idx + i, start_idx + (i+1) % len(coords)])
            if is_hole:
                poly = Polygon(coords)
                rep_pt = poly.representative_point()
                holes_pts.append([rep_pt.x, rep_pt.y])

        add_ring(self.config["outer_boundary"])
        for hole in self.config["holes"]:
            add_ring(hole, is_hole=True)

        poly_dict = {'vertices': vertices, 'segments': segments}
        if holes_pts: poly_dict['holes'] = holes_pts
            
        B = tr.triangulate(poly_dict, 'p')
        triangles_flat = B['triangles'].flatten()
        coords = B['vertices']
        halfedges = build_halfedges(triangles_flat)
        triangles_2d = B['triangles']
        
        wall_edges = set()
        if 'segments' in B:
             seg_set = set(tuple(sorted((s[0], s[1]))) for s in B['segments'])
             for t_idx, tri in enumerate(triangles_2d):
                 for i in range(3):
                     u, v = tri[i], tri[(i+1)%3]
                     if tuple(sorted((u, v))) in seg_set: 
                         wall_edges.add(t_idx * 3 + i)
        self.wall_edges = wall_edges

        num_triangles = len(triangles_flat) // 3
        tri_boxes = []
        for t_idx in range(num_triangles):
            p1, p2, p3 = coords[triangles_flat[t_idx*3 : t_idx*3+3]]
            minx = min(p1[0], p2[0], p3[0]); maxx = max(p1[0], p2[0], p3[0])
            miny = min(p1[1], p2[1], p3[1]); maxy = max(p1[1], p2[1], p3[1])
            tri_boxes.append(box(minx, miny, maxx, maxy))
        tri_strtree = STRtree(tri_boxes)

        # Cache the NavMesh graph!
        G = nx.Graph()
        centroids = {}
        for i in range(num_triangles):
            p_idx = triangles_flat[i*3:i*3+3]
            centroids[i] = np.mean(coords[p_idx], axis=0)
            
        for edg in range(len(halfedges)):
            adj = halfedges[edg]
            if adj != -1 and not self.obstructs(edg):
                t1 = edg // 3; t2 = adj // 3
                G.add_edge(t1, t2, weight=math.dist(centroids[t1], centroids[t2]))
        self.nav_graph = G

        return {'triangles': triangles_flat, 'coords': coords, 'halfedges': halfedges,
                'tri_strtree': tri_strtree, 'tri_boxes': tri_boxes}

    def obstructs(self, edge_idx): 
        return edge_idx in self.wall_edges

    def get_visibility(self, viewpoint):
        return get_visibility_polygon(viewpoint, self.d, self.obstructs)

    def get_frontiers(self, vis_poly):
        boundary = vis_poly.boundary
        if boundary.is_empty: return []
        parts = list(boundary.geoms) if boundary.geom_type == 'MultiLineString' else [boundary]
        frontiers = []
        for part in parts:
            coords = list(part.coords)
            for i in range(len(coords)-1):
                p1, p2 = coords[i], coords[i+1]
                if math.dist(p1, p2) < 1.0: continue
                mx, my = (p1[0] + p2[0])/2, (p1[1] + p2[1])/2
                mid_pt = Point(mx, my)
                if self.boundary_geom.distance(mid_pt) < 0.1: continue
                candidates_idx = self.tree.query(box(mx - 0.1, my - 0.1, mx + 0.1, my + 0.1))
                is_wall = any(self.obstacles[idx].distance(mid_pt) < 0.1 for idx in candidates_idx)
                if not is_wall: frontiers.append(LineString([p1, p2]))
        return frontiers