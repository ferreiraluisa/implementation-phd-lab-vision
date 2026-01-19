import torch
from torch.utils.data import Dataset
import os
from os.path import join, exists
import pickle
import glob
from PIL import Image
import torchvision.transforms as T

class Human36MVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, use_3d=True):
        """
        Args:
            root_dir (str): path to processed Human3.6M folder
            transform (callable, optional): image transformations
            use_3d (bool): whether to return 3D poses
        """
        self.root_dir = root_dir
        self.transform = transform
        self.use_3d = use_3d

        # collect all sequence directories
        self.seq_dirs = glob.glob(join(root_dir, 'S*', '*', 'cam_*'))
        self.data = []

        for seq_dir in self.seq_dirs:
            # load poses
            gt_path = join(seq_dir, 'gt_poses.pkl')
            if not exists(gt_path):
                continue
            with open(gt_path, 'rb') as f:
                poses = pickle.load(f)
            pose2d = poses['2d']
            pose3d = poses['3d'] if use_3d else None

            # collect all frames
            frames = sorted(glob.glob(join(seq_dir, 'frame*.png')))
            n_frames = min(len(frames), len(pose2d))  # match poses length
            for i in range(n_frames):
                self.data.append({
                    'img_path': frames[i],
                    'pose2d': pose2d[i],
                    'pose3d': pose3d[i] if use_3d else None
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        img = Image.open(sample['img_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)  

        pose2d = torch.tensor(sample['pose2d'], dtype=torch.float32)
        pose3d = torch.tensor(sample['pose3d'], dtype=torch.float32) if self.use_3d else None

        if self.use_3d:
            return img, pose2d, pose3d
        else:
            return img, pose2d
