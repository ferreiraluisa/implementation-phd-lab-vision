"""Code adapted from 
https://github.com/akanazawa/human_dynamics/blob/master/src/datasets/h36/read_human36m.py#L449 
BY ME(Luísa Ferreira). 

Preprocesses Human3.6M dataset to extract frames and save along with 3D poses to train PHD model.
"""

from glob import glob
import os
from os import makedirs, system
from os.path import join, getsize, exists
import pickle
from spacepy import pycdf
import sys

import cv2
import numpy as np

from absl import flags


flags.DEFINE_string(
    'source_dir', '/scratch1/storage/human36m_full_raw',
    'Root dir of the original Human3.6M dataset unpacked with metadata.xml'
)
flags.DEFINE_string('out_dir', '/scratch1/storage/human36m_25fps',
                    'Output directory')
flags.DEFINE_integer('frame_skip', 2,
                     'subsample factor, 5 corresponds to 10fps, 2=25fps')

FLAGS = flags.FLAGS

colors = np.random.randint(0, 255, size=(17, 3))
joint_ids = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

# Mapping from H36M joints to LSP joints (0:13). In this roder:
_COMMON_JOINT_IDS = np.array([
    3,  # R ankle
    2,  # R knee
    1,  # R hip
    4,  # L hip
    5,  # L knee
    6,  # L ankle
    16,  # R Wrist
    15,  # R Elbow
    14,  # R shoulder
    11,  # L shoulder
    12,  # L Elbow
    13,  # L Wrist
    8,  # Neck top
    10,  # Head top
])

def read_frames(video_path, n_frames=None):
    """Read frames from video."""
    vid = cv2.VideoCapture(video_path)
    imgs = []
    if n_frames is None:
        n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in range(n_frames):
        success, img = vid.read()
        if not success:
            break
        imgs.append(img)
    return imgs

def read_poses_3d(cdf_path, joint_ids=joint_ids):
    """Read 3D poses from Human3.6M CDF file."""
    data = pycdf.CDF(cdf_path)
    poses = data['Pose'][...][0]  # (N, 64)
    poses3d = [poses[i].reshape(-1, 3)[joint_ids] for i in range(poses.shape[0])]
    return poses3d

def main(raw_data_root, output_root, frame_skip):
    import itertools

    subjects = [1, 5, 6, 7, 8, 9, 11]
    cameras = range(1, 5)
    trials = [1, 2]
    actions = [
        'Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing',
        'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'TakingPhoto',
        'Waiting', 'Walking', 'WalkingDog', 'WalkTogether'
    ]
    
    all_pairs = list(itertools.product(subjects, range(1,16), trials, cameras))

    for subj, action_id, trial, cam in all_pairs:
        action_name = actions[action_id-1]
        seq_name = f"{action_name}_{trial-1}"
        output_dir = join(output_root, f"S{subj}", seq_name, f"cam_{cam-1}")
        if not exists(output_dir):
            makedirs(output_dir)

        # Video path
        video_path = glob(join(raw_data_root, f"S{subj}", "Videos", f"{action_name}.*mp4"))[0]
        # 3D pose path
        pose3d_path = glob(join(raw_data_root, f"S{subj}", "MyPoseFeatures/D3_Positions_mono", f"{action_name}.*cdf"))[0]

        # Read frames and poses
        poses3d = read_poses_3d(pose3d_path)
        imgs = read_frames(video_path)
        imgs = imgs[:len(poses3d)]  # make sure lengths match

        # Subsample
        imgs = imgs[::frame_skip]
        poses3d = poses3d[::frame_skip]

        # Save frames
        for i, img in enumerate(imgs):
            frame_path = join(output_dir, f"frame{i:04d}.png")
            if not exists(frame_path) or getsize(frame_path) == 0:
                cv2.imwrite(frame_path, img)

        # Save only 3D joints
        gt_path = join(output_dir, 'gt_poses.pkl')
        if not exists(gt_path):
            with open(gt_path, 'wb') as f:
                pickle.dump({'3d': poses3d}, f)

        print(f"Processed: S{subj}, {action_name}, trial {trial}, cam {cam}")

if __name__ == '__main__':
    FLAGS(sys.argv)
    main(FLAGS.source_dir, FLAGS.out_dir, FLAGS.frame_skip)