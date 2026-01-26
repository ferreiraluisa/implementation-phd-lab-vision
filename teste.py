from torch.utils.data import DataLoader
from model import PHDFor3DJoints
from dataset import Human36MPreprocessedClips

root = "Human3.6M_preprocessed"

train_ds = Human36MPreprocessedClips(root, split="train", seq_len=40, stride=5, return_meta=False)
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)


video, joints = next(iter(train_dl))
print(video.shape)   # (B,T,3,224,224)
print(joints.shape)  # (B,T,17,3)