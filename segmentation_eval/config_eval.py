import sys
import os

sys.path.append(os.path.split(os.path.dirname(__file__))[0])

stanford_dir = "...my_path/Stanford2D-3DS/"
pano_mask = "segmentation_eval/pano-mask.png"
output_dir = "segmentation_eval/outputs/"

UNet_RGB="graph_networks/pretrained_weights/UNet_2d3ds.pth"
UNet_weights = "graph_networks/pretrained_weights/UNet_2d3ds_withDepth.pth"
UNet_finetuned_sphere = "graph_networks/pretrained_weights/UNet_2d3ds_withDepth_finetuned.pth"
