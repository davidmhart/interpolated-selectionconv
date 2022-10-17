import json
import argparse
import os
from typing import *
#from distconv.point import MakeDist

import config_eval as config  # for setting system path and data directory

import graph_io as gio
import numpy as np
import torch
import torchvision
import tqdm
import utils
from torch_geometric.data import Data
from graph_networks.graph_transforms import transform_network
from PIL import Image
from seg_metrics import accuracy, mean_iou
from torchvision import io
from torchvision.models.segmentation import fcn_resnet50
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize
import numpy as np
import cv2
import clusters as C
from sphere_helpers import equirec2cubic,cubic2equirec
import matplotlib.pyplot as plt
from graph_networks.UNet_graph import UNet as UNet_graph
from graph_networks.UNet import UNet
from segmentation_eval.stanford2d3d import Stanford2D3DDataset

import torchvision.transforms.functional as tvF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def create_masks(network_output_image, num_classes, perform_argmax=True):
    """ creates masks for segmentation outputs

    Parameters
    ----------
    network_output_image: the output image from the segmentation network with shape: [num_classes, height, width] and of type long

    Returns
    -------
    The masks of all the segmentations ready for draw_segmentation_masks with shape: [num_classes, height, width] and of type bool
    """

    if perform_argmax:
        _, H, W = network_output_image.shape
        output_image = torch.argmax(network_output_image, 0)
    else:
        H, W = network_output_image.shape
        output_image = network_output_image
    output_masks = torch.zeros((num_classes, H, W), dtype=torch.bool)
    for i in range(num_classes):
        output_masks[i] = output_image == i
    return output_masks



def get_graph(image,image_type,depth=6,device='cpu'):
    
    if image_type == "2d":
        graph,metadata = gio.image2Graph(image,depth=depth,device=device)
    elif image_type == "panorama":
        graph,metadata = gio.panorama2Graph(image,depth=depth,device=device)
    elif image_type == "cubemap":
        graph,metadata = gio.sphere2Graph_cubemap(image,depth=depth,device=device,face_size=192)
    elif image_type == "superpixel":
        graph,metadata = gio.superpixel2Graph(image,depth=depth,device=device)
    elif image_type == "sphere":
        graph,metadata = gio.sphere2Graph(image,depth=depth,device=device,scale=.75)
    else:
        raise ValueError(f"image_type not known: {image_type}")

    return graph,metadata

def get_x(image,image_type,device='cpu'):
    
    if image_type == "2d":
        x = gio.image2Graph(image,mask=None,x_only=True,device=device)
    elif image_type == "panorama":
        x = gio.panorama2Graph(image,mask=mask,x_only=True,device=device)
    elif image_type == "cubemap":
        x = gio.sphere2Graph_cubemap(image,mask=mask,x_only=True,device=device,face_size=192)
    elif image_type == "superpixel":
        x = gio.superpixel2Graph(image,mask=mask,x_only=True,device=device)
    elif image_type == "sphere":
        x = gio.sphere2Graph(image,mask=mask,x_only=True,device=device,scale=.75)
    else:
        raise ValueError(f"image_type not known: {image_type}")

    return x

def project_graph(x,metadata,image_type):
    
    if image_type == "cubemap":
        # Put back into equirectangular form
        result_image = gio.graph2Sphere_cubemap(x,metadata)
    elif image_type == "superpixel":
        # Paint back in superpixel segments
        result_image = gio.graph2Superpixel(x,metadata)
    elif image_type == "sphere":
        # Interpolate Point Cloud
        result_image = gio.graph2Sphere(x,metadata)
    else:
        result_image = gio.graph2Image(x,metadata)

    return result_image



def eval_segmentation( index_image, gt, mask, n_classes, weighting=None, full_res=True):
    if full_res:
        index_image = resize(index_image, (mask.shape[1], mask.shape[2]), InterpolationMode.BILINEAR)
    index_image = torch.argmax(index_image, 0)
    mask = mask.squeeze()
    pred = index_image[torch.where(mask)].flatten().long()
    act = gt[torch.where(mask)].flatten().long()
    if weighting is not None:
        weighting = weighting[torch.where(mask)].flatten()    
    iou = mean_iou(pred, act, n_classes, weighting)
    acc = accuracy(pred, act, weighting)
    return iou, acc


def rgb_transform(rgb):
    if rgb.shape[0] != 3 and rgb.shape[0] != 4:
        raise ValueError(f"rgb's shape is not rgb or rgb-d: {rgb.shape}")

    # RGB and Depth
    rgb = rgb.float()
    rgb[:3] = rgb[:3]/255
    rgb[:3] = tvF.normalize(rgb[:3], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    if rgb.shape[0] == 4:
        # Depth information
        rgb[3] = torch.clamp(rgb[3],0,4) - 2 # Normalize depth

    return rgb

def val_transforms(im, seg):
    im = resize(im, (512, 1024))
    seg = resize(seg.unsqueeze(0), (512, 1024), InterpolationMode.NEAREST).squeeze(0)
    im = rgb_transform(im)
    #im = im.float() / 255
    #im = normalize(im, IMAGENET_MEAN, IMAGENET_STD)
    return im, seg

def val_transforms_full_res(im, seg):
    im = resize(im, (512, 1024))
    #seg = resize(seg.unsqueeze(0), (512, 1024), InterpolationMode.NEAREST).squeeze(0)
    im = rgb_transform(im)
    #im = im.float() / 255
    #im = normalize(im, IMAGENET_MEAN, IMAGENET_STD)
    return im, seg

def shifted(im):
    c, h, w = im.shape
    im = torch.cat([im[:, :, w//2:], im[:, :, :w//2]], 2)
    return im

def save_segments(model, image_type, image, gt, mask, n_classes, outname):
    og_image = image * torch.tensor(gio.IMAGENET_STD).view(3, 1, 1) + torch.tensor(gio.IMAGENET_MEAN).view(3, 1, 1)
    og_image = (og_image * 255).byte()
    seg = segment(model, image_type, image, n_classes)
    masks = create_masks(seg, n_classes)
    masks = masks & mask
    segmentation = torchvision.utils.draw_segmentation_masks(og_image, masks)
    io.write_png(segmentation, f"{outname}-pred.png")
    io.write_png(shifted(segmentation), f"{outname}-pred-shifted.png")

    masks = create_masks(gt, n_classes, False)
    masks = masks & mask
    segmentation = torchvision.utils.draw_segmentation_masks(og_image, masks)
    io.write_png(segmentation, f"{outname}-gt.png")
    io.write_png(shifted(segmentation), f"{outname}-gt-shifted.png")
    io.write_png(og_image, f"{outname}-og.png")
    io.write_png(shifted(og_image), f"{outname}-og-shifted.png")




def main(input_im,image_type, device) -> torch.Tensor:

    im = utils.loadImage(input_im)

    # Get initial graph structure    
    graph,metadata = get_graph(im,image_type,device=device)
    network = UNet_graph(14,in_channels=4)
    network.load_state_dict(torch.load(config.UNet_finetuned))

    network = network.to(device)
    network.eval()
        

    with torch.no_grad():
        outputs = network(graph)
    index_image = project_graph(outputs,metadata,image_type)

    index_image = torch.tensor(index_image,dtype=torch.float).permute((2,0,1))
    
    final_image = create_masks(index_image,14)
    
    print(final_image.shape)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_im")
    parser.add_argument("image_type") # Literal["2d","panorama","cubemap","sphere","superpixel", "vanilla", "vanilla_cubemap"]
    parser.add_argument("device")
    args = parser.parse_args()
    main(args.input_im,args.image_type,args.device)
