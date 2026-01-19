"""Code adapted from
https://github.com/akanazawa/human_dynamics/blob/master/src/datasets/h36/read_human36m.py#L449
BY ME(Luísa Ferreira).

Preprocesses Human3.6M dataset to extract frames and save along with 3D poses to train PHD model.
OPTIMIZED VERSION
"""

from glob import glob
import os
from os import makedirs
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

joint_ids = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

def read_frames_optimized(video_path, frame_skip, n_frames=None):
    """Read frames from video with subsampling during read."""
    vid = cv2.VideoCapture(video_path)
    imgs = []
    
    if n_frames is None:
        n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_idx = 0
    while frame_idx < n_frames:
        vid.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, img = vid.read()
        if not success:
            break
        imgs.append(img)
        frame_idx += frame_skip
    
    vid.release()
    return imgs

def read_poses_3d(cdf_path, joint_ids=joint_ids, frame_skip=1):
    """Read 3D poses from Human3.6M CDF file with subsampling."""
    data = pycdf.CDF(cdf_path)
    poses = data['Pose'][::frame_skip, 0]  # Subsample directly
    poses3d = poses.reshape(poses.shape[0], -1, 3)[:, joint_ids]
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

    all_pairs = list(itertools.product(subjects, range(1, 16), trials, cameras))

    for subj, action_id, trial, cam in all_pairs:
        action_name = actions[action_id - 1]
        seq_name = f"{action_name}_{trial - 1}"
        output_dir = join(output_root, f"S{subj}", seq_name, f"cam_{cam - 1}")
        
        gt_path = join(output_dir, 'gt_poses.pkl')
        
        # Skip se já processado
        if exists(gt_path):
            print(f"Skipping (already done): S{subj}, {action_name}, trial {trial}, cam {cam}")
            continue
        
        makedirs(output_dir, exist_ok=True)

        # Video path
        video_path = glob(join(raw_data_root, f"S{subj}", "Videos", f"{action_name}.*mp4"))[0]
        # 3D pose path
        pose3d_path = glob(join(raw_data_root, f"S{subj}", "MyPoseFeatures/D3_Positions_mono", f"{action_name}.*cdf"))[0]

        # Read poses with subsampling
        poses3d = read_poses_3d(pose3d_path, frame_skip=frame_skip)
        
        # Read frames with subsampling
        imgs = read_frames_optimized(video_path, frame_skip, n_frames=len(poses3d) * frame_skip)
        
        # Ajustar tamanhos
        min_len = min(len(imgs), len(poses3d))
        imgs = imgs[:min_len]
        poses3d = poses3d[:min_len]

        # Save frames com JPEG 
        for i, img in enumerate(imgs):
            frame_path = join(output_dir, f"frame{i:04d}.jpg")
            cv2.imwrite(frame_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Save 3D joints
        with open(gt_path, 'wb') as f:
            pickle.dump({'3d': poses3d}, f)

        print(f"Processed: S{subj}, {action_name}, trial {trial}, cam {cam}")

if __name__ == '__main__':
    FLAGS(sys.argv)
    main(FLAGS.source_dir, FLAGS.out_dir, FLAGS.frame_skip)