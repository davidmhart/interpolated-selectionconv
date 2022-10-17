import config

import argparse
import utils
import graph_io as gio
from mesh_helpers import loadMesh
from clusters import *
from tqdm import tqdm,trange

from graph_networks.LinearStyleTransfer_vgg import encoder,decoder
from graph_networks.LinearStyleTransfer_matrix import TransformLayer

from graph_networks.LinearStyleTransfer.libs.Matrix import MulLayer
from graph_networks.LinearStyleTransfer.libs.models import encoder4, decoder4

import matplotlib.pyplot as plt

from mesh_config import mesh_info


def styletransfer(mesh_name,style_path,device,mesh_method,N):
    
    info = mesh_info[mesh_name]
    
    load_directory = info["load_directory"]
    mesh_fn = info["mesh_fn"]
    texture_fn = info["texture_fn"]
    save_directory = info["save_directory"]
    content_ref = utils.loadImage(load_directory + texture_fn)
    style_ref = utils.loadImage(style_path, shape=(256,256))
    
    mesh = loadMesh(load_directory + mesh_fn)
    
    if mesh_method == "texture":
        content,content_meta = gio.texture2Graph(content_ref,mesh,depth=3,device=device)
    elif mesh_method == "texture3D":
        content,content_meta = gio.texture2Graph_3D(content_ref,mesh,depth=3,device=device)
    elif mesh_method == "sampling":
        content,content_meta = gio.mesh2Graph(content_ref,mesh,N=N,depth=3,device=device)
    else:
        raise ValueError(f"mesh_method not known: {mesh_method}")

    style,_ = gio.image2Graph(style_ref,depth=3,device=device)

    # Load original network
    enc_ref = encoder4()
    dec_ref = decoder4()
    matrix_ref = MulLayer('r41')

    enc_ref.load_state_dict(torch.load('graph_networks/LinearStyleTransfer/models/vgg_r41.pth'))
    dec_ref.load_state_dict(torch.load('graph_networks/LinearStyleTransfer/models/dec_r41.pth'))
    matrix_ref.load_state_dict(torch.load('graph_networks/LinearStyleTransfer/models/r41.pth'))

    # Copy weights to graph network
    enc = encoder(padding_mode="replicate")
    dec = decoder(padding_mode="replicate")
    matrix = TransformLayer()

    with torch.no_grad():
        enc.copy_weights(enc_ref)
        dec.copy_weights(dec_ref)
        matrix.copy_weights(matrix_ref)

    #content = content.to(device)
    #style = style.to(device)
    enc = enc.to(device)
    dec = dec.to(device)
    matrix = matrix.to(device)

    # Run graph network
    with torch.no_grad():
        cF = enc(content)
        sF = enc(style)
        feature,transmatrix = matrix(cF['r41'],sF['r41'],
                                     content.edge_indexes[3],content.selections_list[3],
                                     style.edge_indexes[3],style.selections_list[3],
                                     content.interps_list[3] if hasattr(content,'interps_list') else None)
        result = dec(feature,content)
        result = result.clamp(0,1)
        
    # Save/show result
    if mesh_method == "texture" or mesh_method == "texture3D":
        # Remake Texture
        result_image = gio.graph2Texture(result,content_meta,view3D=True)
    elif mesh_method == "sampling":
        # Interpolate Point Cloud
        result_image = gio.graph2Mesh(result,content_meta,view3D=True)
    else:
        raise ValueError(f"mesh_method not known: {mesh_method}")

    '''
    pos2D = content_meta.uvs.cpu().numpy()
    pos2D[:,0] = pos2D[:,0]*content_meta.original.shape[1]
    pos2D[:,1] = 1-pos2D[:,1] # UV puts y=0 at the bottom
    pos2D[:,1] = pos2D[:,1]*content_meta.original.shape[0]
       
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(pos2D[:,0],pos2D[:,1],c=result.cpu().numpy());
    ax2.scatter(pos2D[:,0],pos2D[:,1],c=content.x.cpu().numpy());
    plt.show()
    '''
    
    plt.imsave(save_directory + texture_fn,result_image)
    plt.imshow(result_image);plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mesh_name",
        type=str,
    )
    parser.add_argument(
        "style_path",
        type=str,
        default="style_ims/style0.jpg"
    )
    parser.add_argument(
        "--device",
        default= 0 if torch.cuda.is_available() else "cpu",
        choices=list(range(torch.cuda.device_count())) + ["cpu"] or ["cpu"]
    )
    parser.add_argument(
        "--mesh_method",
        default="sampling"
    )
    parser.add_argument(
        "--points",
        "-N",
        type=int,
        default=100000
    )

    args = parser.parse_args()
    styletransfer(**vars(args))


if __name__ == "__main__":
    main()
