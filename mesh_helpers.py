import trimesh
import torch
import numpy as np

def loadMesh(mesh_fn):
    mesh = trimesh.load(mesh_fn, force="mesh")
    #mesh = trimesh.load_mesh(mesh_fn)
    return mesh    

def getUVs(mesh):
    if (isinstance(mesh.visual,trimesh.visual.color.ColorVisuals)):
        uvs = mesh.visual.to_texture().uv
    else:
        uvs = mesh.visual.uv
        
    return uvs

def setTexture(mesh,texture):
    from PIL import Image
    if texture.dtype == np.float32 or texture.dtype == np.float64:
        texture = (255*texture).astype(np.uint8)
    im = Image.fromarray(texture)
    tex = trimesh.visual.TextureVisuals(uv=getUVs(mesh),image=im)
    new_mesh=trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, visual=tex, validate=True, process=True)
    return new_mesh

def sampleSurface(mesh,N,return_x=False,device='cpu'):
    result = trimesh.sample.sample_surface(mesh,N,face_weight=None,sample_color=return_x)
    pos3D = result[0]
    face_indexes = result[1]
    if return_x:
        x = result[2][:,:3]/255 # No alpha channels, between 0-1
    
    # Determine normals (the same as face normals)
    normals = mesh.face_normals[face_indexes]
    
    if return_x:
    
        # Determine UVs associated with 3D points
        vertices_all = mesh.vertices
        uvs_all = getUVs(mesh)
        faces = mesh.faces[face_indexes]

        a = vertices_all[faces[:,0]]
        b = vertices_all[faces[:,1]]
        c = vertices_all[faces[:,2]]

        w0, w1, w2 = getBarycentricWeights(pos3D,a,b,c,use_numpy=True)

        w0 = np.expand_dims(w0,axis=1)
        w1 = np.expand_dims(w1,axis=1)
        w2 = np.expand_dims(w2,axis=1)

        uvs = w0 * uvs_all[faces[:,0]] + w1 * uvs_all[faces[:,1]] + w2 * uvs_all[faces[:,2]]

        uvs = torch.tensor(uvs,dtype=torch.float).to(device)
        
    # Return as tensor
    pos3D = torch.tensor(pos3D,dtype=torch.float).to(device)
    normals = torch.tensor(normals,dtype=torch.float).to(device)
    
    if return_x:
        x = torch.tensor(x,dtype=torch.float).to(device)
        return pos3D,normals,uvs,x
    else:
        return pos3D,normals

def getBarycentricWeights(p,a,b,c,use_numpy=False):
    
    # Taken from https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
    v0 = b-a
    v1 = c-a
    v2 = p-a
    
    if use_numpy:
        d00 = np.sum(v0*v0,axis=1)
        d01 = np.sum(v0*v1,axis=1)
        d11 = np.sum(v1*v1,axis=1)
        d20 = np.sum(v2*v0,axis=1)
        d21 = np.sum(v2*v1,axis=1)
    else:
        d00 = torch.sum(v0*v0,dim=1)
        d01 = torch.sum(v0*v1,dim=1)
        d11 = torch.sum(v1*v1,dim=1)
        d20 = torch.sum(v2*v0,dim=1)
        d21 = torch.sum(v2*v1,dim=1)
    denom = d00*d11 - d01*d01
    w1 = (d11 * d20 - d01 * d21)/denom
    w2 = (d00 * d21 - d01 * d20)/denom
    
    w0 = 1 - w1 - w2
    
    return w0,w1,w2