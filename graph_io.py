import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn

import graph_helpers as gh
import sphere_helpers as sh
import mesh_helpers as mh
import texture_helpers as th
import clusters as cl
import utils

from torch_scatter import scatter
#from skimage.segmentation import slic
from fast_slic import Slic

import math
from math import pi, sqrt

from warnings import warn

def image2Graph(data, gt = None, mask = None, depth = 1, x_only = False, device = 'cpu'):
    
    _,ch,rows,cols = data.shape
    
    x = torch.reshape(data,(ch,rows*cols)).permute((1,0)).to(device)
    
    if mask is not None:
        # Mask out nodes
        node_mask = torch.where(mask.flatten())
        x = x[node_mask]
    
    if gt is not None:
        y = gt.flatten().to(device)
        if mask is not None:
            y = y[node_mask]
    
    if x_only:
        if gt is not None:
            return x,y
        else:
            return x
    
    im_pos = gh.getImPos(rows,cols)
    
    if mask is not None:
        im_pos = im_pos[node_mask]
    
    # Make "point cloud" for clustering
    pos2D = gh.convertImPos(im_pos,flip_y=False)
    
    # Generate initial graph
    edge_index = gh.grid2Edges(pos2D)
    directions = pos2D[edge_index[1]] - pos2D[edge_index[0]]
    selections = gh.edges2Selections(edge_index,directions,interpolated=False,y_down=True)
    
    # Generate info for downsampled versions of the graph
    clusters, edge_indexes, selections_list = cl.makeImageClusters(pos2D,cols,rows,edge_index,selections,depth=depth,device=device)
    
    # Make final graph and metadata needed for mapping the result after going through the network
    graph = Data(x=x,clusters=clusters,edge_indexes=edge_indexes,selections_list=selections_list,interps_list=None)
    metadata = Data(original=data,im_pos=im_pos.long(),rows=rows,cols=cols,ch=ch)
    
    if gt is not None:
        graph.y = y
    
    return graph,metadata

def graph2Image(result,metadata,canvas=None):
    
    x = utils.toNumpy(result,permute=False)
    im_pos = utils.toNumpy(metadata.im_pos,permute=False)
    if canvas is None:
        canvas = utils.makeCanvas(x,metadata.original)
    
    # Paint over the original image (neccesary for masked images)
    canvas[im_pos[:,0],im_pos[:,1]] = x
    
    return canvas

def panorama2Graph(data, gt=None, mask = None, depth = 1, x_only = False, device = 'cpu'):
    # A simple graph structure that connects the left and right sides of an image
    # Also illustrates a more direct edge connection method
    
    _,ch,rows,cols = data.shape
    
    x = torch.reshape(data,(ch,rows*cols)).permute((1,0)).to(device)
    
    if mask is not None:
        mask = mask.flatten()
        x = gh.maskNodes(mask,x)
    
    if gt is not None:
        y = torch.reshape(gt,(ch,rows*cols)).permute((1,0)).to(device)
        if mask is not None:
            y = y[node_mask]
    
    if x_only:
        if gt is not None:
            return x,y
        else:
            return x
    
    im_pos = gh.getImPos(rows,cols)
    if mask is not None:
        im_pos = gh.maskNodes(mask,im_pos)
    
    # Get the initial edges of the graph
    index_img = torch.arange(rows*cols).reshape((rows,cols))

    sources = []
    targets = []
    selections = []
    
    gh.makeEdges(sources,targets,selections,index_img[:,:-1], index_img[:,1:], 1)
    gh.makeEdges(sources,targets,selections,index_img[:-1,:], index_img[1:,:], 7)
    gh.makeEdges(sources,targets,selections,index_img[:-1,:-1], index_img[1:,1:], 8)
    gh.makeEdges(sources,targets,selections,index_img[:-1,1:], index_img[1:,:-1], 6)
    gh.makeEdges(sources,targets,selections,index_img, index_img, 0, reverse=False)
    gh.makeEdges(sources,targets,selections,index_img[:,-1], index_img[:,0], 1)
    
    edge_index = torch.row_stack((torch.tensor(sources,dtype=torch.long),torch.tensor(targets,dtype=torch.long)))
    selections = torch.tensor(selections,dtype=torch.long)
    
    # Mask graph if needed
    if mask is not None:
        edge_index,selections = gh.maskGraph(mask,edge_index,selections)
    
    # Generate info for downsampled versions of the graph
    clusters, edge_indexes, selections_list = cl.makeImageClusters(gh.convertImPos(im_pos),cols,rows,edge_index,selections,depth=depth,device=device)
    
    # Make final graph and metadata needed for mapping the result after going through the network    
    graph = Data(x=x,clusters=clusters,edge_indexes=edge_indexes,selections_list=selections_list,interps_list=None)
    metadata = Data(original=data,im_pos=im_pos.long(),rows=rows,cols=cols,ch=ch)
    
    if gt is not None:
        graph.y = y
    
    return graph,metadata

def graph2Panorama(result,metadata):
    return graph2Image(result,metadata)

def sphere2Graph_cubemap(data, gt=None, mask = None, depth = 1, x_only = False, device = 'cpu', face_size = None):
    
    equirec_image = utils.toNumpy(data)
    equi_rows,equi_cols,_ = equirec_image.shape
    
    cubemap = sh.equirec2cubic(equirec_image, face_size)
    
    if mask is not None:
        # Convert mask to RGB image before passing in
        mask_r = utils.toNumpy(mask.squeeze(),permute=False).astype(np.float32)
        mask_rgb = np.stack((mask_r,mask_r,mask_r),axis=2)
        mask_cubemap = sh.equirec2cubic(mask_rgb,face_size)
        mask_cubemap = mask_cubemap[:,:,0] > 0.5
    
        #import matplotlib.pyplot as plt
        #plt.imshow(mask_cubemap);plt.show()
 
    total_rows, total_cols, ch = cubemap.shape
    
    # Size of each face
    rows = total_rows//3
    cols = total_cols//4
    
    if rows != cols:
        warn("Warning: Not perfect squares in cube map. Resulting graph may have errors")
    
    # Build x
    horiz_faces = cubemap[rows:2*rows]
    top_faces = cubemap[:rows,cols:2*cols]
    bottom_faces = cubemap[2*rows:,cols:2*cols]
    
    x = np.vstack((np.reshape(horiz_faces,(-1,ch)),np.reshape(top_faces,(-1,ch)),np.reshape(bottom_faces,(-1,ch))))
    
    if mask is not None:
        horiz_mask = mask_cubemap[rows:2*rows].flatten()
        top_mask = mask_cubemap[:rows,cols:2*cols].flatten()
        bottom_mask = mask_cubemap[2*rows:,cols:2*cols].flatten()
        
        np_mask = np.concatenate((horiz_mask,top_mask,bottom_mask))
        x = x[np.where(np_mask)[0]]
        
    x = torch.tensor(x,dtype=torch.float).to(device)
    
    if gt is not None:
        gt_r = utils.toNumpy(gt.squeeze(),permute=False).astype(np.float32)
        gt_rgb = np.stack((gt_r,gt_r,gt_r),axis=2)
        y_cubemap = sh.equirec2cubic(gt_rgb, face_size)
        y_cubemap = y_cubemap[:,:,0].astype(np.uint8)
        
        # Build y
        y_horiz_faces = y_cubemap[rows:2*rows].flatten()
        y_top_faces = y_cubemap[:rows,cols:2*cols].flatten()
        y_bottom_faces = y_cubemap[2*rows:,cols:2*cols].flatten()
    
        y = np.concatenate((y_horiz_faces,y_top_faces,y_bottom_faces))
    
        if mask is not None:
            y = y[np.where(np_mask)[0]]
        
        y = torch.tensor(y,dtype=torch.float).to(device)
    
    if x_only:
        if gt is not None:
            return x,y
        else:
            return x
    
    # Build im_pos
    horiz_im_pos = gh.getImPos(rows,4*cols,rows)
    top_im_pos = gh.getImPos(rows,cols,0,cols)
    bottom_im_pos = gh.getImPos(rows,cols,2*rows,cols)
    im_pos = torch.vstack((horiz_im_pos,top_im_pos,bottom_im_pos))
    
    # Build graph
    horiz_nodes = torch.arange(rows*4*cols).reshape((rows,4*cols))
    top_nodes = torch.arange(rows*4*cols,rows*5*cols).reshape((rows,cols))
    bottom_nodes = torch.arange(rows*5*cols,rows*6*cols).reshape((rows,cols))
    
    edge_index, selections = sh.buildCubemapEdges(horiz_nodes,top_nodes,bottom_nodes)

    if mask is not None:
        mask = torch.tensor(np_mask,dtype=torch.bool)
        edge_index,selections = gh.maskGraph(mask,edge_index,selections)
    
    # Generate info for downsampled versions of the graph
    clusters, edge_indexes, selections_list = cl.makeCubemapClusters(gh.convertImPos(im_pos,flip_y=False),total_cols,total_rows,edge_index,selections,depth=depth,device=device,mask=mask)
    
    if mask is not None:
        im_pos = gh.maskNodes(mask,im_pos) # Must mask after to avoid issues with cluster method
    
    # Make final graph and metadata needed for mapping the result after going through the network    
    graph = Data(x=x,clusters=clusters,edge_indexes=edge_indexes,selections_list=selections_list,interps_list=None)
    metadata = Data(original=cubemap,im_pos=im_pos.long(),rows=equi_rows,cols=equi_cols,ch=ch)
    
    if gt is not None:
        graph.y = y
    
    return graph,metadata
    
def graph2Sphere_cubemap(result,metadata):
    x = utils.toNumpy(result,permute=False)
    im_pos = utils.toNumpy(metadata.im_pos,permute=False)
    canvas = utils.makeCanvas(x,metadata.original)
    
    # Paint over the original image (neccesary for masked images)
    #for i in range(len(im_pos)):
    #    canvas[im_pos[i][0],im_pos[i][1]] = x[i]
    canvas[im_pos[:,0],im_pos[:,1]] = x
        
    if canvas.ndim < 3:
        canvas = np.expand_dims(canvas,axis=2)
        
    # Convert back to equirec
    return np.squeeze(sh.cubic2equirec(canvas,metadata.rows,metadata.cols))

def texture2Graph(data, mesh, mask = None, depth = 1, x_only = False, device = 'cpu'):
    
    if mask is not None:
        warn("Masks are not currently implemented for texture graphs")
    
    image = utils.toNumpy(data)
    rows,cols,ch = image.shape
    
    mask, boundaries, lookup = th.seperateTexture(mesh,rows,cols,return_lookup=True)
    
    mask_torch = torch.tensor(mask,dtype=torch.bool).flatten()
    
    # Calculate masked node data
    x = np.reshape(image,(rows*cols,ch))
    x = torch.tensor(x,dtype=torch.float)
    x = gh.maskNodes(mask_torch,x)
    x = x.to(device)
    
    if x_only:
        return x

    im_pos = gh.getImPos(rows,cols)
    im_pos = gh.maskNodes(mask_torch,im_pos)
    
    # Build initial graph
    edge_index, selections = th.buildTextureEdges(mask,boundaries,lookup,mesh,rows,cols)
    
    # Generate info for downsampled versions of the graph
    clusters, edge_indexes, selections_list = cl.makeImageClusters(gh.convertImPos(im_pos,flip_y=False),cols,rows,edge_index,selections,depth=depth,device=device)
    
    # Make final graph and metadata needed for mapping the result after going through the network
    graph = Data(x=x,clusters=clusters,edge_indexes=edge_indexes,selections_list=selections_list,interps_list=None)
    metadata = Data(original=data,im_pos=im_pos.long(),mesh=mesh,mask=mask,rows=rows,cols=cols,ch=ch)
    
    return graph,metadata

def graph2Texture(result,metadata,dilations=1,view3D=False):
    canvas = graph2Image(result,metadata)
    
    # Dilate around the texture edges to account for bleed across boundaries
    canvas,_ = th.textureDilation(canvas,metadata.mask,dilations)
    canvas = np.clip(canvas,0,1)
    
    if view3D:
        mesh = mh.setTexture(metadata.mesh,canvas)
        mesh.show()
    
    return canvas

def superpixel2Graph(data, downsample=8, sigma=10, mask = None, depth = 1, x_only = False, device = 'cpu'):
    
    if mask is not None:
        warn("Masks are not currently implemented for superpixels")
    
    image = utils.toNumpy(data)
    rows, cols, ch = image.shape

    num_pix = image.size//(downsample**2)

    #from time import time
    #start = time()

    #Code for non-fast slic
    #segments = slic(image,n_segments = num_pix, sigma = sigma, start_label=0)
    
    slic_fast = Slic(num_components=num_pix, compactness=sigma, min_size_factor=0)
    segments = slic_fast.iterate((image*255).astype(np.uint8))
    
    #print("SLIC:", time() - start)

    #from skimage.segmentation import mark_boundaries
    #plt.imshow(mark_boundaries(image,segments));plt.show()

    segments = torch.tensor(segments.flatten(), dtype=torch.long)

    original_x = np.reshape(image,(rows*cols,ch))
    original_x = torch.tensor(original_x, dtype=torch.float)

    x = scatter(original_x, segments, dim=0, reduce='mean').to(device)
    
    if x_only:
        return x
    
    # Make "point cloud" for clustering
    im_pos = gh.getImPos(rows,cols)
    pos2D = gh.convertImPos(im_pos,flip_y=False)
    
    pos2D = scatter(pos2D, segments, dim=0, reduce='mean')
    
    # Generate initial graph
    edge_index = gh.knn2Edges(pos2D,knn=12)
    directions = pos2D[edge_index[1]] - pos2D[edge_index[0]]
    selections = gh.edges2Selections(edge_index,directions,interpolated=False,y_down=True)
    
    # Simplify the graph to have a single selection in each direction
    edge_index, selections = gh.simplifyGraph(edge_index,selections,torch.linalg.norm(directions,dim=1))
    
    # Generate info for downsampled versions of the graph
    clusters, edge_indexes, selections_list = cl.makeImageClusters(pos2D,cols,rows,edge_index,selections,depth=depth,device=device)
    
    # Make final graph and metadata needed for mapping the result after going through the network
    graph = Data(x=x,clusters=clusters,edge_indexes=edge_indexes,selections_list=selections_list,interps_list=None)
    metadata = Data(original=data,im_pos=im_pos.long(),segments=segments,rows=rows,cols=cols,ch=ch)
    
    return graph,metadata

def graph2Superpixel(result,metadata):
    
    x = utils.toNumpy(result,permute=False)
    im_pos = utils.toNumpy(metadata.im_pos,permute=False)
    segments = utils.toNumpy(metadata.segments,permute=False)
    canvas = utils.makeCanvas(x,metadata.original)

    canvas[im_pos[:,0],im_pos[:,1]] = x[segments]

    return canvas

### Begin Interpolated Methods ###

def sphere2Graph(data, structure="layering", cluster_method="layering", scale=1.0, stride=2, interpolation_mode = "angle", gt = None, mask = None, depth = 1, x_only = False, device = 'cpu'):
    
    _,ch,rows,cols = data.shape
    
    if structure == "equirec":
        # Use the original data to start with
        cartesian, spherical = sh.sampleSphere_Equirec(scale*rows,scale*cols)
    elif structure == "layering":
        cartesian, spherical = sh.sampleSphere_Layering(scale*rows)
    elif structure == "spiral":
        cartesian, spherical = sh.sampleSphere_Spiral(scale*rows,scale*cols)
    elif structure == "icosphere":
        cartesian, spherical = sh.sampleSphere_Icosphere(scale*rows)
    elif structure == "random":
        cartesian, spherical = sh.sampleSphere_Random(scale*rows,scale*cols)
    else:
        raise ValueError("Sphere structure unknown")
    
    if interpolation_mode == "bary":            
        bary_d = pi/(scale*rows)
    else:
        bary_d = None
    
    # Get the landing point for each node
    sample_x, sample_y = sh.spherical2equirec(spherical[:,0],spherical[:,1],rows,cols)
    
    if mask is not None:

        node_mask = gh.maskPoints(mask,sample_x,sample_y)
        sample_x = sample_x[node_mask]
        sample_y = sample_y[node_mask]
        spherical = spherical[node_mask]
        cartesian = cartesian[node_mask]
    
    features = utils.bilinear_interpolate(data, sample_x, sample_y).to(device)
    
    if gt is not None:
        features_y = utils.bilinear_interpolate(gt.unsqueeze(0), sample_x, sample_y).to(device)
    
    if x_only:
        if gt is not None:
            return features,features_y
        else:
            return features
        
    # Build initial graph
    edge_index,directions = gh.surface2Edges(cartesian,cartesian)
    edge_index,selections,interps = gh.edges2Selections(edge_index,directions,interpolated=True,bary_d=bary_d)
    
    # Generate info for downsampled versions of the graph
    clusters, edge_indexes, selections_list, interps_list = cl.makeSphereClusters(cartesian,edge_index,selections,interps,rows*scale,cols*scale,cluster_method,stride=stride,bary_d=bary_d,depth=depth,device=device)
    
    # Make final graph and metadata needed for mapping the result after going through the network
    graph = Data(x=features,clusters=clusters,edge_indexes=edge_indexes,selections_list=selections_list,interps_list=interps_list)
    metadata = Data(original=data,pos3D=cartesian,mask=mask,rows=rows,cols=cols,ch=ch)

    if gt is not None:
        graph.y = features_y   
    
    return graph, metadata
    
def graph2Sphere(features,metadata):
    
    # Generate equirectangular points and their 3D locations
    theta, phi = sh.equirec2spherical(metadata.rows, metadata.cols)
    x,y,z = sh.spherical2xyz(theta,phi)
    
    v = torch.stack((x,y,z),dim=1)
    
    # Find closest 3D point to each equirectangular point
    nearest = torch.reshape(knn(metadata.pos3D,v,3)[1],(len(v),3))
    
    #Interpolate based on proximty to each node
    w0 = 1/torch.linalg.norm((v - metadata.pos3D[nearest[:,0]]),dim=1, keepdim=True).to(features.device)
    w1 = 1/torch.linalg.norm((v - metadata.pos3D[nearest[:,1]]),dim=1, keepdim=True).to(features.device)
    w2 = 1/torch.linalg.norm((v - metadata.pos3D[nearest[:,2]]),dim=1, keepdim=True).to(features.device)
    
    w0 = torch.nan_to_num(w0, nan=1e6)
    w1 = torch.nan_to_num(w1, nan=1e6)
    w2 = torch.nan_to_num(w2, nan=1e6)
    
    w0 = torch.clamp(w0,0,1e6)
    w1 = torch.clamp(w1,0,1e6)
    w2 = torch.clamp(w2,0,1e6)
    
    total = w0 + w1 + w2
    
    #w0,w1,w2 = mh.getBarycentricWeights(v,metadata.pos3D[nearest[:,0]],metadata.pos3D[nearest[:,1]],metadata.pos3D[nearest[:,2]])
    
    #w0 = w0.unsqueeze(1).to(features.device)
    #w1 = w1.unsqueeze(1).to(features.device)
    #w2 = w2.unsqueeze(1).to(features.device)
        
    result = (w0*features[nearest[:,0]] + w1*features[nearest[:,1]] + w2*features[nearest[:,2]])/total
    
    #result = result.clamp(0,1)

    if hasattr(metadata,"mask"):
        mask = utils.toNumpy(metadata.mask.squeeze(),permute=False)
        canvas = utils.makeCanvas(result,metadata.original)
        result = np.reshape(result.data.cpu().numpy(),(metadata.rows,metadata.cols,features.shape[1]))
        canvas[np.where(mask)] = result[np.where(mask)]
        return canvas
    else:
        return np.reshape(result.data.cpu().numpy(),(metadata.rows,metadata.cols,features.shape[1]))

def texture2Graph_3D(data, mesh, up_vector = None, mask = None, depth = 1, x_only = False, device = 'cpu'):
    """ Use full point cloud to determine edge indices """

    if up_vector == None:
        up_vector = 2*torch.rand((1,3))-1
        up_vector = up_vector/torch.linalg.norm(up_vector,dim=1)

    if mask is not None:
        warn("Masks are not currently implemented for texture graphs")
    
    image = utils.toNumpy(data)
    rows,cols,ch = image.shape
    
    mask, _ = th.seperateTexture(mesh,rows,cols)
        
    mask_torch = torch.tensor(mask,dtype=torch.bool).flatten()
    
    # Calculate masked node data
    x = np.reshape(image,(rows*cols,ch))
    x = torch.tensor(x,dtype=torch.float)
    x = gh.maskNodes(mask_torch,x)
    x = x.to(device)
    
    print(len(x))
    
    if x_only:
        return x
    
    im_pos = gh.getImPos(rows,cols)
    im_pos = gh.maskNodes(mask_torch,im_pos)
    
    # Build point cloud from texture data
    pos3D,normals = th.texture2Points3D(mask,mesh)
    
    # Build initial graph
    edge_index,directions = gh.surface2Edges(pos3D,normals,up_vector)
    edge_index,selections,interps = gh.edges2Selections(edge_index,directions,interpolated=True)

    # Generate info for downsampled versions of the graph
    clusters, edge_indexes, selections_list, interps_list = cl.makeUVClusters(pos3D,mask,mesh,edge_index,selections,interps,up_vector=up_vector,depth=depth,device=device)
    
    # Make final graph and metadata needed for mapping the result after going through the network
    graph = Data(x=x,clusters=clusters,edge_indexes=edge_indexes,selections_list=selections_list,interps_list=interps_list)
    metadata = Data(original=data,im_pos=im_pos.long(),mesh=mesh,mask=mask,rows=rows,cols=cols,ch=ch)
    
    return graph,metadata

def graph2Texture_3D(result,metadata,view3D=False):
    return graph2Texture(result,metadata,view3D)

def mesh2Graph(data, mesh, up_vector = None, N = 100000, ratio=.25, mask = None, depth = 1, x_only = False, device = 'cpu'):
    """ Sample mesh faces to determine graph """

    if up_vector == None:
        up_vector = torch.tensor([[1,1,1]],dtype=torch.float)
        #up_vector = 2*torch.rand((1,3))-1
        up_vector = up_vector/torch.linalg.norm(up_vector,dim=1)

    if mask is not None:
        warn("Masks are not currently implemented for mesh graphs")      
    
    pos3D, normals, uvs, x = mh.sampleSurface(mesh,N,return_x=True)
    
    x = x.to(device)
    
    if x_only:
        warn("x_only returns randomly selected points for mesh2Graph. Do not use with previous graph structures")
        return x
    
    # Build initial graph
    edge_index,directions = gh.surface2Edges(pos3D,normals,up_vector,k_neighbors=16)
    edge_index,selections,interps = gh.edges2Selections(edge_index,directions,interpolated=True)

    # Generate info for downsampled versions of the graph
    clusters, edge_indexes, selections_list, interps_list = cl.makeSurfaceClusters(pos3D,normals,edge_index,selections,interps,ratio=ratio,up_vector=up_vector,depth=depth,device=device)
    #clusters, edge_indexes, selections_list, interps_list = cl.makeMeshClusters(pos3D,mesh,edge_index,selections,interps,ratio=ratio,up_vector=up_vector,depth=depth,device=device)
    
    # Make final graph and metadata needed for mapping the result after going through the network
    graph = Data(x=x,clusters=clusters,edge_indexes=edge_indexes,selections_list=selections_list,interps_list=interps_list)
    metadata = Data(original=data,pos3D=pos3D,uvs=uvs,mesh=mesh)
    
    return graph,metadata

def graph2Mesh(features,metadata,view3D=False):
    
    features = features.cpu().numpy()
    
    canvas = utils.toNumpy(metadata.original)
    rows,cols,ch = canvas.shape
    
    # Get 2D positions by scaling uv
    pos2D = metadata.uvs.cpu().numpy()
    pos2D[:,0] = pos2D[:,0]*cols
    pos2D[:,1] = 1-pos2D[:,1] # UV puts y=0 at the bottom
    pos2D[:,1] = pos2D[:,1]*rows
    
    # Generate desired points
    row_space = np.arange(rows)
    col_space = np.arange(cols)
    col_image,row_image = np.meshgrid(col_space,row_space)
    
    canvas = utils.interpolatePointCloud2D(pos2D,features,col_image,row_image)
    canvas = np.clip(canvas,0,1)

    if view3D:
        mesh = mh.setTexture(metadata.mesh,canvas)
        mesh.show()
    
    return canvas
