import numpy as np
import open3d as o3d
import os
import urllib.request
import gzip
import tarfile
import shutil
import time
import pdb
from scipy.spatial.transform import Rotation as Rot

def create_plane(center, normal, plane_size, axis_size = 0.03):
    mesh = o3d.TriangleMesh()
    quat = np.r_[normal,0] 

    e = np.array([0,1,0])
    cross_vec_1 = np.cross(normal,e)
    if(cross_vec_1[0] < 0):
        cross_vec_1 *= -1
    cross_vec_1 /= np.linalg.norm(cross_vec_1)

    cross_vec_2 = np.cross(normal,cross_vec_1)
    if(cross_vec_2[2] < 0):
        cross_vec_2 *= -1
    cross_vec_2 /= np.linalg.norm(cross_vec_2)
    
    vec_set_basis = np.array([[0,1,0],[1,0,0],[0,0,1]])
    vec_set_normal = np.array([normal, cross_vec_1, cross_vec_2])
    rr = Rot.match_vectors(vec_set_basis, vec_set_normal)[0]
    transform_mat = np.eye(4)
    transform_mat[:3,:3] = rr.as_dcm().T
    transform_mat[:3,3] = center

    p1 = np.array([-plane_size*0.5, 0, -plane_size*0.5])
    p2 = np.array([plane_size*0.5, 0, -plane_size*0.5])
    p3 = np.array([plane_size*0.5, 0, plane_size*0.5])
    p4 = np.array([-plane_size*0.5, 0, plane_size*0.5])
    _mesh_points = np.array([p1,p2,p3,p4], dtype=np.float)
    mesh.vertices = o3d.Vector3dVector(_mesh_points)
    _mesh_triangles = np.array([[0, 2, 1], [2, 0, 3]])
    _mesh_triangles = np.concatenate([_mesh_triangles, _mesh_triangles[:,::-1]], axis=0)
    mesh.triangles = o3d.Vector3iVector(_mesh_triangles)
    mesh.paint_uniform_color([1, 0.706, 0])    
    mesh.compute_vertex_normals()
    mesh.transform(transform_mat)

    mesh_cylinder = o3d.create_mesh_cylinder(radius=axis_size/50, height=axis_size)
    mesh_cylinder.paint_uniform_color([0, 1.0, 0])   
    transform_yaw = np.array([[1,0,0,0],[0,0,1,axis_size/2],[0,1,0,0],[0,0,0,1]])
    mesh_cylinder.transform(transform_yaw)
    mesh_cylinder.transform(transform_mat)

    return mesh, mesh_cylinder

def ransac_plane(pcd, n_iter=100, dthres_inlier=0.010, cut_percentile=0.8):
    pcd_ary = np.asarray(pcd.points)
    n_point = pcd_ary.shape[0]
    rand_0 = np.random.randint(n_point, size=n_iter)
    rand_1 = np.random.randint(n_point-1, size=n_iter)
    rand_2 = np.random.randint(n_point-2, size=n_iter)

    idx_0 = rand_0
    idx_1 = idx_0 + 1 + rand_1
    idx_2 = idx_0 + 1 + rand_2
    idx_2[idx_2 == idx_1] += 1
    idx_1[idx_1 > n_point-1] -= n_point
    idx_2[idx_2 > n_point-1] -= n_point

    point_0 = pcd_ary[idx_0]
    point_1 = pcd_ary[idx_1]
    point_2 = pcd_ary[idx_2]

    origin = point_0
    v_1 = point_1 - origin
    v_2 = point_2 - origin

    normal = np.cross(v_2, v_1)
    norm_normal = np.linalg.norm(normal, axis=-1)
    bflg_zeronorm = (norm_normal == 0.)
    norm_normal[bflg_zeronorm] = 1.
    normal /= norm_normal.reshape(-1, 1)
    normal[bflg_zeronorm] = [0., 1., 0.]

    dist2plane = np.absolute(((pcd_ary.reshape(1,-1,3) - origin.reshape(-1,1,3)) * normal.reshape(-1,1,3)).sum(-1))
    is_inlier = (dist2plane < dthres_inlier)
    n_inlier = is_inlier.sum(-1) - 3
    frac_inlier = n_inlier.astype(np.float32) / float(n_point-3)

    n_inlier_for_denom = n_inlier.astype(np.float32)
    n_inlier_for_denom[n_inlier == 0] = 1.
    dist_mean = (dist2plane * is_inlier).sum(axis=-1) / n_inlier_for_denom
    cand_err = dist_mean

    cut_variable = frac_inlier

    n_cand = cand_err.shape[0]
    cut_thres = np.sort(cut_variable)[int(n_cand * cut_percentile)]
    flg_pass = (cut_variable >= cut_thres)
    idx_sorted = np.argsort(cand_err)
    i_best = idx_sorted[np.argmax(flg_pass[idx_sorted])]
    pdb.set_trace()

    best_origin = origin[i_best]
    best_normal = normal[i_best]
    if best_normal[1] < 0.:
        best_normal *= -1.

    return best_origin, best_normal

pcd = o3d.read_point_cloud("./plane_subsampled.pcd")
origin, normal = ransac_plane(pcd, n_iter=100, dthres_inlier=0.005, cut_percentile=0.5)
origin = np.mean(np.asarray(pcd.points),axis=0)
rr = Rot.from_euler("yxz",[0, 10e-5, 0],degrees=True)
rr2 = Rot.from_euler("yxz",[30,0,0],degrees=True)
axis_normal = rr2.as_dcm().dot(rr.as_dcm().dot(np.array([0,1,0])))
e_vec = np.array([1,0,0])
#v = np.cos(-60/180*np.pi) * axis_normal + np.sin(-60/180*np.pi)*np.cross(axis_normal,e_vec)+(1-np.cos(-60/180*np.pi))*np.dot(axis_normal,e_vec)*e_vec #rodrigues

#mesh, mesh_axis  = create_plane(np.array([0,0,0]), axis_normal, 0.10, axis_size=0.03)
mesh, mesh_axis  = create_plane(origin, normal, plane_size = 0.20, axis_size=0.03)
o3d.draw_geometries([mesh, mesh_axis,pcd])

