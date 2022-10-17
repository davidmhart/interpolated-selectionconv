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
#from torch_geometric.utils import accuracy, mean_iou
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



def get_graph(image,sampling,clustering,interpolation,mask,depth=6,device='cpu'):
    
    graph,metadata = gio.sphere2Graph(image,structure=sampling,cluster_method=clustering,
                                      interpolation_mode=interpolation,mask=mask,depth=depth,
                                      device=device,scale=.75)

    return graph,metadata

def get_x(image,sampling,clustering,mask,device='cpu'):
    
    x = gio.sphere2Graph(image,structure=sampling,cluster_method=clustering,
                         mask=mask,x_only=True,device=device,scale=.75)

    return x

def project_graph(x,metadata):
    
    # Interpolate Point Cloud
    result_image = gio.graph2Sphere(x,metadata)

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




def main(sampling,clustering,interpolation_mode,device, full_res=True) -> torch.Tensor:
    mask_original = torchvision.io.read_image(config.pano_mask)
    mask = resize(mask_original, (512, 1024), InterpolationMode.NEAREST).type(torch.bool)
    mask_original = mask_original.type(torch.bool)
    # mask = np.where(~mask.numpy(), 255, 0).astype(np.uint8)
    _, val_split = Stanford2D3DDataset.get_splits(1)
    if full_res:
        dataset = Stanford2D3DDataset(config.stanford_dir, datatype="pano", with_depth=True, areas=val_split, transforms=val_transforms_full_res)
    else:
        dataset = Stanford2D3DDataset(config.stanford_dir, datatype="pano", with_depht=True, areas=val_split, transforms=val_transforms)
    
    #reference_network = fcn_resnet50(pretrained=False, num_classes=dataset.num_classes)
    #reference_network.load_state_dict(torch.load(config.weights))
    reference_network = UNet(dataset.num_classes,in_channels=4)
    reference_network.load_state_dict(torch.load(config.UNet_weights))
    
    # Get initial graph structure    
    graph,metadata = get_graph(dataset[0][0].unsqueeze(0),sampling,clustering,interpolation_mode,mask,device=device)
    network = UNet_graph(dataset.num_classes,in_channels=4)
    with torch.no_grad():
        network.copy_weights(reference_network) 

    network = network.to(device)
    network.eval()
    
    if full_res:
        weighting = torch.tensor(utils.cosineWeighting(2048,4096),dtype=torch.float)
    else:
        weighting = torch.tensor(utils.cosineWeighting(512,1024),dtype=torch.float)
    
    best_iou_i = 0
    best_iou = -np.inf
    best_acc_i = 0
    best_acc = -np.inf
    ious = []
    accs = []
    

    #for i in tqdm.trange(10):
    for i in tqdm.trange(len(dataset)):
        im, gt = dataset[i]

        im = im.unsqueeze(0)

        graph.x = get_x(im,sampling,clustering,mask,device=device)
        with torch.no_grad():
            outputs = network(graph)
        index_image = project_graph(outputs,metadata)

        index_image = torch.tensor(index_image,dtype=torch.float).permute((2,0,1))

        if full_res:
            iou, acc = eval_segmentation(index_image, gt, mask_original, dataset.num_classes, weighting, full_res)
        else:
            iou, acc = eval_segmentation(index_image, gt, mask, dataset.num_classes, weighting, full_res)
        ious.append(iou.item())
        accs.append(acc)
        if iou > best_iou:
            best_iou = iou
            best_iou_i = i
        if acc > best_acc:
            best_acc = acc
            best_acc_i = i
                
                
    best_iou_im, best_iou_gt = dataset[best_iou_i]
    #save_segments(graph_network, image_type, best_iou_im, best_iou_gt, mask, dataset.num_classes, config.output_dir+f"{image_type}-best_iou")
    best_acc_im, best_acc_gt = dataset[best_acc_i]
    #save_segments(graph_network, image_type, best_acc_im, best_acc_gt, mask, dataset.num_classes, config.output_dir+f"{image_type}-best_acc")
    print(f"Mean IOU: {np.mean(ious)}")
    #print(f"Best iou IDX: {best_iou_i}")
    #print(f"Best IOU: {best_iou}")
    print(f"Mean ACC: {np.mean(accs)}")
    #print(f"Best acc IDX: {best_acc_i}")
    #print(f"Best Pixel Level Accuracy: {best_acc}")
    #with open(f"segmentation_eval/{image_type}-out.json", "w") as file:
    #    json.dump({
    #        "iou": ious,
    #        "acc": accs
    #    }, file)
        # tqdm.tqdm.write(f"  {i}: IOU: {ious:.6f} ACC: {acc:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sampling")
    parser.add_argument("clustering")
    parser.add_argument("interpolation")
    parser.add_argument("device")
    args = parser.parse_args()
    main(args.sampling,args.clustering,args.interpolation,args.device)
