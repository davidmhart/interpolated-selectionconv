import os
import torch
import json
from PIL import Image
from torchvision import io
import numpy as np

def get_subroot(root):
    return os.path.join(root, "buildingparser", "noXYZ")


def get_datapath(subroot, area, datatype, task):
    return os.path.join(subroot, f"area_{area}", datatype, task)


def get_sorted_paths(directory, ext):
    if isinstance(ext, str):
        ext = (ext,)
    filenames = sorted(os.listdir(directory))
    return [os.path.join(directory, name) for name in filenames if os.path.splitext(name)[-1] in ext]


def get_index( color ):
    ''' Parse a color as a base-256 number and returns the index
    Args:
        color: A 3-tuple in RGB-order where each element \in [0, 255]
    Returns:
        index: an int containing the indec specified in 'color'
    '''
    return color[0] * 256 * 256 + color[1] * 256 + color[2]


def parse_label( label ):
    """ Parses a label into a dict """
    res = {}
    clazz, instance_num, room_type, room_num, area_num = label.split( "_" )
    res[ 'instance_class' ] = clazz
    res[ 'instance_num' ] = int( instance_num )
    res[ 'room_type' ] = room_type
    res[ 'room_num' ] = int( room_num )
    res[ 'area_num' ] = int( area_num )
    return res


class Stanford2D3DDataset(torch.utils.data.Dataset):
    classes = ["<UNK>", "ceiling", "floor", "wall", "beam", "column", "window", "door", "table", "chair", "sofa", "bookcase", "board", "clutter"]
    num_classes = len(classes)
    @classmethod
    def get_splits(clss, fold_num):
        folds = {
            1: (("1", "2", "3", "4", "6"), ("5a", "5b")),
            2: (("1", "3", "5a", "5b", "6"), ("2", "4")),
            3: (("2", "4", "5a", "5b"), ("1", "3", "6"))
        }
        return folds[fold_num]

    def __init__(
            self,
            root,
            task="semantic",
            areas=("1", "2", "3", "4", "5a", "5b", "6"),
            datatype="data",
            rgb_load_func=lambda path: io.read_image(path, io.ImageReadMode.RGB),
            seg_load_func=lambda path: torch.tensor(np.array(Image.open(path))),
            rgb_transform=None,
            seg_transform=None,
            transforms=None,
            with_depth=False,
        ):
        self.root = root
        self.subroot = get_subroot(root)
        self.rgb_load_func = rgb_load_func
        self.seg_load_func = seg_load_func
        self.task = task
        self.transforms = transforms
        self.areas = areas
        self.with_depth = with_depth
        self.rgb_paths = []
        self.seg_paths = []
        self.depth_paths = []
        with open(os.path.join(self.subroot, "assets", "semantic_labels.json")) as file:
            semantic_labels = json.load(file)
        self.clss_indices = []
        for i, label in enumerate(semantic_labels):
            label = parse_label(label)
            instance_class = label["instance_class"]
            index = self.classes.index(instance_class)
            self.clss_indices.append(index)
        self.clss_indices = torch.tensor(self.clss_indices)
        for area in areas:
            seg_root = get_datapath(self.subroot, area, datatype, "semantic")
            rgb_root = get_datapath(self.subroot, area, datatype, "rgb")
            depth_root = get_datapath(self.subroot, area, datatype, "depth")
            rgb_paths = get_sorted_paths(rgb_root, ".png")
            seg_paths = get_sorted_paths(seg_root, ".png")
            depth_paths = get_sorted_paths(depth_root, ".png")
            self.rgb_paths.extend(rgb_paths)
            self.seg_paths.extend(seg_paths)
            self.depth_paths.extend(depth_paths)
        self.get_y = {
            "depth": self.get_depth,
            "semantic": self.get_seg,
        }

    def get_image(self, idx):
        rgb_path = self.rgb_paths[idx]
        rgb = self.rgb_load_func(rgb_path)
        return rgb

    def get_seg(self, idx):
        seg_path = self.seg_paths[idx]
        seg = self.seg_load_func(seg_path).long().permute(2, 0, 1)
        seg[:, (seg[0] == 13) & (seg[1] == 13) & (seg[2] == 13)] = 0
        index_seg = get_index(seg)
        index_seg = self.clss_indices[index_seg]
        return index_seg

    def get_depth(self, idx):
        depth_path = self.depth_paths[idx]
        depth = torch.tensor(np.array(Image.open(depth_path)),dtype=torch.float)/512
        return depth

    def __getitem__(self, idx):
        
        try:
            x = self.get_image(idx)
            y = self.get_y[self.task](idx)

            if self.with_depth:
                depth = self.get_depth(idx)
                x = torch.cat((x,depth.unsqueeze(0)),dim=0)
            if self.transforms:
                x, y = self.transforms(x, y)
            return x, y
        except:
            print("Had trouble with",idx)
            print("Filename:",self.rgb_paths[idx])
            quit()

    def __len__(self):
        return len(self.rgb_paths)
