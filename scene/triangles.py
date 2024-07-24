import torch
import open3d as o3d
import numpy as np
from arguments import ModelParams
from torch import nn
from scene.gaussian_model import GaussianModel
import os

class Triangle:
    def add_vertex(self, vertex):
        if len(vertex) != 3:
            raise ValueError("Vertex must have three coordinates")
        self.vertices.append(vertex)
    def get_facets(self):
        return self.facets
    def create_from_ply(self,args : ModelParams,ply_path):
        '''
        pcd=o3d.io.read_point_cloud(ply_path)
        
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 1.5 * avg_dist

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([radius, radius * 2]))
        '''
        
        #alpha_shape
        '''
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        alpha = avg_dist * 10  # Adjust alpha to control mesh detail

        # Perform Alpha Shapes reconstruction
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=alpha)
        '''
        
        pcd = o3d.io.read_point_cloud(ply_path)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # 使用泊松重建算法生成网格
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        
        # 过滤掉低密度的三角形
        density_threshold = 4
        vertices_to_remove = [i for i, density in enumerate(densities) if density < density_threshold]
        mesh.remove_vertices_by_index(vertices_to_remove)
        
        #print(densities)
        
        trian = np.asarray(mesh.triangles)
        vert = np.asarray(mesh.vertices)
        
        self.facets = torch.tensor(trian)
        if self.facets.dtype == torch.long:
            self.facets = self.facets.to(torch.int32)
            
        self.verts=torch.tensor(vert)
        if self.verts.dtype == torch.long:
            self.verts = self.verts.to(torch.int32)
        
        print("self.facets")
        print(self.facets.shape)
        ply_path1=os.path.join(args.source_path, "sparse/0/mesh.ply")
        o3d.io.write_triangle_mesh(ply_path1, mesh)
    def load_ply(self,ply_path):
        mesh = o3d.io.read_triangle_mesh(ply_path)
        trian = np.asarray(mesh.triangles)
        vert = np.asarray(mesh.vertices)
        
        self.facets = torch.tensor(trian)
        if self.facets.dtype == torch.long:
            self.facets = self.facets.to(torch.int32)
        
        self.verts=torch.tensor(vert)
        if self.verts.dtype == torch.long:
            self.verts = self.verts.to(torch.int32)
    def save(self,path,gaussians : GaussianModel):
        mesh = o3d.geometry.TriangleMesh()
        self.verts=gaussians.get_xyz.detach().cpu()
        mesh.vertices = o3d.utility.Vector3dVector(self.verts.numpy())
        mesh.triangles = o3d.utility.Vector3iVector(self.facets.numpy())
        o3d.io.write_triangle_mesh(path, mesh)