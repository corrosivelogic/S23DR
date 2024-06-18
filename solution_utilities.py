# Description: This file contains the handcrafted solution for the task of wireframe reconstruction 

import io
from PIL import Image as PImage
import numpy as np
from collections import defaultdict
import cv2
from typing import Tuple, List
from scipy.spatial.distance import cdist

from hoho.read_write_colmap import read_cameras_binary, read_images_binary, read_points3D_binary
from hoho.color_mappings import gestalt_color_mapping, ade20k_color_mapping


def empty_solution():
    '''Return a minimal valid solution, i.e. 2 vertices and 1 edge.'''
    return np.zeros((2,3)), [(0, 1)]
    

def convert_entry_to_human_readable(entry):
    out = {}
    already_good = ['__key__', 'wf_vertices', 'wf_edges', 'edge_semantics', 'mesh_vertices', 'mesh_faces', 'face_semantics', 'K', 'R', 't']
    for k, v in entry.items():
        if k in already_good:
            out[k] = v
            continue
        if k == 'points3d':
            out[k] = read_points3D_binary(fid=io.BytesIO(v))
        if k == 'cameras':
            out[k] = read_cameras_binary(fid=io.BytesIO(v))
        if k == 'images':
            out[k] = read_images_binary(fid=io.BytesIO(v))
        if k in ['ade20k', 'gestalt']:
            out[k] =  [PImage.open(io.BytesIO(x)).convert('RGB') for x in v]
        if k == 'depthcm':
            out[k] = [PImage.open(io.BytesIO(x)) for x in entry['depthcm']]
    return out


def get_vertices_and_edges_from_segmentation(gest_seg_np, edge_th = 50.0):
    '''Get the vertices and edges from the gestalt segmentation mask of the house'''
    vertices = []
    connections = []
    # Apex
    apex_color = np.array(gestalt_color_mapping['apex'])
    apex_mask = cv2.inRange(gest_seg_np,  apex_color-0.5, apex_color+0.5)
    if apex_mask.sum() > 0:
        output = cv2.connectedComponentsWithStats(apex_mask, 8, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        stats, centroids = stats[1:], centroids[1:]
        
        for i in range(numLabels-1):
            vert = {"xy": centroids[i], "type": "apex"}
            vertices.append(vert)
    
    eave_end_color = np.array(gestalt_color_mapping['eave_end_point'])
    eave_end_mask = cv2.inRange(gest_seg_np,  eave_end_color-0.5, eave_end_color+0.5)
    if eave_end_mask.sum() > 0:
        output = cv2.connectedComponentsWithStats(eave_end_mask, 8, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        stats, centroids = stats[1:], centroids[1:]
        
        for i in range(numLabels-1):
            vert = {"xy": centroids[i], "type": "eave_end_point"}
            vertices.append(vert) 
    # Connectivity
    apex_pts = []
    apex_pts_idxs = []
    for j, v in enumerate(vertices):
        apex_pts.append(v['xy'])
        apex_pts_idxs.append(j)
    apex_pts = np.array(apex_pts)
            
    # Ridge connects two apex points
    for edge_class in ['eave', 'ridge', 'rake', 'valley']:
        edge_color = np.array(gestalt_color_mapping[edge_class])
        mask = cv2.morphologyEx(cv2.inRange(gest_seg_np,
                                            edge_color-0.5,
                                            edge_color+0.5),
                                cv2.MORPH_DILATE, np.ones((11, 11)))
        line_img = np.copy(gest_seg_np) * 0
        if mask.sum() > 0:
            output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
            (numLabels, labels, stats, centroids) = output
            stats, centroids = stats[1:], centroids[1:]
            edges = []
            for i in range(1, numLabels):
                y,x = np.where(labels == i)
                xleft_idx = np.argmin(x)
                x_left = x[xleft_idx]
                y_left = y[xleft_idx]
                xright_idx = np.argmax(x)
                x_right = x[xright_idx]
                y_right = y[xright_idx]
                edges.append((x_left, y_left, x_right, y_right))
                cv2.line(line_img, (x_left, y_left), (x_right, y_right), (255, 255, 255), 2)
            edges = np.array(edges)
            if (len(apex_pts) < 2) or len(edges) <1:
                continue
            pts_to_edges_dist = np.minimum(cdist(apex_pts, edges[:,:2]), cdist(apex_pts, edges[:,2:]))
            connectivity_mask = pts_to_edges_dist <= edge_th
            edge_connects = connectivity_mask.sum(axis=0)
            for edge_idx, edgesum in enumerate(edge_connects):
                if edgesum>=2:
                    connected_verts = np.where(connectivity_mask[:,edge_idx])[0]
                    for a_i, a in enumerate(connected_verts):
                        for b in connected_verts[a_i+1:]:
                            connections.append((a, b))
    return vertices, connections

def get_uv_depth(vertices, depth):
    '''Get the depth of the vertices from the depth image'''
    uv = []
    for v in vertices:
        uv.append(v['xy'])
    uv = np.array(uv)
    uv_int = uv.astype(np.int32)
    H, W = depth.shape[:2]
    uv_int[:, 0] = np.clip( uv_int[:, 0], 0, W-1)
    uv_int[:, 1] = np.clip( uv_int[:, 1], 0, H-1)
    vertex_depth = depth[(uv_int[:, 1] , uv_int[:, 0])]
    return uv, vertex_depth


def merge_vertices_3d(vert_edge_per_image, th=0.1):
    '''Merge vertices that are close to each other in 3D space and are of same types'''
    all_3d_vertices = []
    connections_3d = []
    all_indexes = []
    cur_start = 0
    types = []
    for cimg_idx, (vertices, connections, vertices_3d) in vert_edge_per_image.items():
        types += [int(v['type']=='apex') for v in vertices]
        all_3d_vertices.append(vertices_3d)
        connections_3d+=[(x+cur_start,y+cur_start) for (x,y) in connections]
        cur_start+=len(vertices_3d)
    all_3d_vertices = np.concatenate(all_3d_vertices, axis=0)
    #print (connections_3d)
    distmat = cdist(all_3d_vertices, all_3d_vertices)
    types = np.array(types).reshape(-1,1)
    same_types = cdist(types, types)
    mask_to_merge = (distmat <= th) & (same_types==0)
    new_vertices = []
    new_connections = []
    to_merge = sorted(list(set([tuple(a.nonzero()[0].tolist()) for a in mask_to_merge])))
    to_merge_final = defaultdict(list)
    for i in range(len(all_3d_vertices)):
        for j in to_merge:
            if i in j:
                to_merge_final[i]+=j
    for k, v in to_merge_final.items():
        to_merge_final[k] = list(set(v))
    already_there = set() 
    merged = []
    for k, v in to_merge_final.items():
        if k in already_there:
            continue
        merged.append(v)
        for vv in v:
            already_there.add(vv)
    old_idx_to_new = {}
    count=0
    for idxs in merged:
        new_vertices.append(all_3d_vertices[idxs].mean(axis=0))
        for idx in idxs:
            old_idx_to_new[idx] = count
        count +=1
    #print (connections_3d)
    new_vertices=np.array(new_vertices)
    #print (connections_3d)
    for conn in connections_3d:
        new_con = sorted((old_idx_to_new[conn[0]], old_idx_to_new[conn[1]]))
        if new_con[0] == new_con[1]:
            continue
        if new_con not in new_connections:
            new_connections.append(new_con)
    #print (f'{len(new_vertices)} left after merging {len(all_3d_vertices)} with {th=}')
    return new_vertices, new_connections

def prune_not_connected(all_3d_vertices, connections_3d):
    '''Prune vertices that are not connected to any other vertex'''
    connected = defaultdict(list)
    for c in connections_3d:
        connected[c[0]].append(c)
        connected[c[1]].append(c)
    new_indexes = {}
    new_verts = []
    connected_out = []
    for k,v in connected.items():
        vert = all_3d_vertices[k]
        if tuple(vert) not in new_verts:
            new_verts.append(tuple(vert))
            new_indexes[k]=len(new_verts) -1
    for k,v in connected.items():
        for vv in v:
            connected_out.append((new_indexes[vv[0]],new_indexes[vv[1]]))
    connected_out=list(set(connected_out))                   
    
    return np.array(new_verts), connected_out


def predict(entry, visualize=False) -> Tuple[np.ndarray, List[int]]:
    good_entry = convert_entry_to_human_readable(entry)
    vert_edge_per_image = {}
    for i, (gest, depth, K, R, t) in enumerate(zip(good_entry['gestalt'],
                                                good_entry['depthcm'], 
                                                good_entry['K'],
                                                good_entry['R'],
                                                good_entry['t'] 
                                                )):
        gest_seg = gest.resize(depth.size)
        gest_seg_np = np.array(gest_seg).astype(np.uint8)
        # Metric3D
        depth_np = np.array(depth) / 2.5 # 2.5 is the scale estimation coefficient
        vertices, connections = get_vertices_and_edges_from_segmentation(gest_seg_np, edge_th = 20.)
        if (len(vertices) < 2) or (len(connections) < 1):
            print (f'Not enough vertices or connections in image {i}')
            vert_edge_per_image[i] = np.empty((0, 2)), [], np.empty((0, 3))
            continue
        uv, depth_vert = get_uv_depth(vertices, depth_np)
        # Normalize the uv to the camera intrinsics
        xy_local = np.ones((len(uv), 3))
        xy_local[:, 0] = (uv[:, 0] - K[0,2]) / K[0,0]
        xy_local[:, 1] = (uv[:, 1] - K[1,2]) / K[1,1]
        # Get the 3D vertices
        vertices_3d_local = depth_vert[...,None] * (xy_local/np.linalg.norm(xy_local, axis=1)[...,None])
        world_to_cam = np.eye(4)
        world_to_cam[:3, :3] = R
        world_to_cam[:3, 3] = t.reshape(-1)
        cam_to_world =  np.linalg.inv(world_to_cam)
        vertices_3d = cv2.transform(cv2.convertPointsToHomogeneous(vertices_3d_local), cam_to_world)
        vertices_3d = cv2.convertPointsFromHomogeneous(vertices_3d).reshape(-1, 3)
        vert_edge_per_image[i] = vertices, connections, vertices_3d
    all_3d_vertices, connections_3d = merge_vertices_3d(vert_edge_per_image, 3.0)
    all_3d_vertices_clean, connections_3d_clean  = prune_not_connected(all_3d_vertices, connections_3d)
    if (len(all_3d_vertices_clean) < 2) or len(connections_3d_clean) < 1:
        print (f'Not enough vertices or connections in the 3D vertices')
        return (good_entry['__key__'], *empty_solution())
    if visualize:
        from hoho.viz3d import plot_estimate_and_gt
        plot_estimate_and_gt(   all_3d_vertices_clean, 
                                connections_3d_clean, 
                                good_entry['wf_vertices'],
                                good_entry['wf_edges'])
    return good_entry['__key__'], all_3d_vertices_clean, connections_3d_clean 
