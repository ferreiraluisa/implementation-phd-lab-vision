import torch
from torch.utils.data import Dataset
import os
from os.path import join, exists
import pickle
from PIL import Image
import torchvision.transforms as T

class Human36MVideoDataset(Dataset):
    def __init__(self, root_dir="Human3.6_preprocessed", subjects=[1,5,6,7,8], cams=[0], 
                 seq_len=40, stride=10, transform=None):
        """
        Args:
            root_dir (str): Root folder of preprocessed Human3.6M data.
            subjects (list): List of subject IDs to include (training subjects).
            cams (list): List of camera indices to use (0-3).
            seq_len (int): Length of each training subsequence.
            stride (int): Stride for sliding windows.
            transform (callable): Transform to apply to frames.
        """
        self.samples = []
        self.seq_len = seq_len
        self.stride = stride
        self.transform = transform
        
        actions = [
                'Directions','Discussion','Eating','Greeting','Phoning','Posing',
                'Purchases','Sitting','SittingDown','Smoking','TakingPhoto',
                'Waiting','Walking','WalkingDog','WalkTogether'
            ]
        
        # Iterate over subjects, actions, cameras
        for subj in subjects:
            for action in actions:
                for cam in cams:
                    seq_dir = join(root_dir, f"S{subj}", f"{action}_0", f"cam_{cam}")
                    gt_path = join(seq_dir, 'gt_poses.pkl')
                    if not exists(gt_path):
                        continue
                    with open(gt_path, 'rb') as f:
                        joints3d = pickle.load(f)['3d']
                    num_frames = len(joints3d)
                    frame_paths = [join(seq_dir, f"frame{i:04d}.png") for i in range(num_frames)]
                    
                    # Sliding window
                    for start in range(0, num_frames - seq_len + 1, stride):
                        end = start + seq_len
                        self.samples.append({
                            'frames': frame_paths[start:end],
                            'joints': joints3d[start:end]
                        })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = []
        for fpath in sample['frames']:
            img = Image.open(fpath).convert('RGB')
            if self.transform:
                img = self.transform(img)
            else:
                img = T.ToTensor()(img)
            frames.append(img)
        frames = torch.stack(frames)  # shape: (seq_len, 3, H, W)
        joints = torch.tensor(sample['joints'], dtype=torch.float32)  # shape: (seq_len, 17, 3)
        return frames, joints
