"""Code adapted from
https://github.com/akanazawa/human_dynamics/blob/master/src/datasets/h36/read_human36m.py#L449
BY ME(Luísa Ferreira).

Preprocesses Human3.6M dataset to extract frames and save along with 3D poses to train PHD model.
OPTIMIZED VERSION WITH MULTIPROCESSING
"""

from glob import glob
import os
from os import makedirs
from os.path import join, exists
import pickle
from spacepy import pycdf
import sys
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from absl import flags

flags.DEFINE_string('source_dir', '/scratch1/storage/human36m_full_raw')
flags.DEFINE_string('out_dir', '/scratch1/storage/human36m_25fps')
flags.DEFINE_integer('frame_skip', 2)
flags.DEFINE_integer('num_workers', None)

FLAGS = flags.FLAGS

joint_ids = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

def read_frames_optimized(video_path, frame_skip, n_frames=None):
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

def read_poses_3d(cdf_path, joint_ids, frame_skip):
    data = pycdf.CDF(cdf_path)
    poses = data['Pose'][::frame_skip, 0]
    poses = poses.reshape(poses.shape[0], -1, 3)
    return poses[:, joint_ids]

def process_sequence(args):
    subj, action_name, trial, cam, raw_data_root, output_root, frame_skip = args

    seq_name = f"{action_name}_{trial - 1}"
    output_dir = join(output_root, f"S{subj}", seq_name, f"cam_{cam - 1}")
    gt_path = join(output_dir, 'gt_poses.pkl')

    if exists(gt_path):
        return f"SKIP S{subj} {action_name} t{trial} c{cam}"

    makedirs(output_dir, exist_ok=True)

    video_files = sorted(glob(join(raw_data_root, f"S{subj}", "Videos", f"{action_name}.*mp4")))
    if len(video_files) < cam:
        return f"ERROR video S{subj} {action_name} cam{cam}"
    video_path = video_files[cam - 1]

    pose_files = sorted(glob(join(raw_data_root, f"S{subj}", "MyPoseFeatures/D3_Positions_mono", f"{action_name}.*cdf")))
    if len(pose_files) < cam:
        return f"ERROR pose S{subj} {action_name} cam{cam}"
    pose_path = pose_files[cam - 1]

    poses3d = read_poses_3d(pose_path, joint_ids, frame_skip)
    imgs = read_frames_optimized(video_path, frame_skip, n_frames=len(poses3d) * frame_skip)

    min_len = min(len(imgs), len(poses3d))
    if min_len == 0:
        return f"ERROR empty S{subj} {action_name} cam{cam}"

    imgs = imgs[:min_len]
    poses3d = poses3d[:min_len]

    for i, img in enumerate(imgs):
        cv2.imwrite(join(output_dir, f"frame{i:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    with open(gt_path, 'wb') as f:
        pickle.dump({'3d': poses3d}, f)

    return f"OK S{subj} {action_name} t{trial} c{cam} {min_len}"

def main(raw_data_root, output_root, frame_skip, num_workers):
    import itertools

    subjects = [1, 5, 6, 7, 8, 9, 11]
    cameras = range(1, 5)
    trials = [1, 2]
    actions = [
        'Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing',
        'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'TakingPhoto',
        'Waiting', 'Walking', 'WalkingDog', 'WalkTogether'
    ]

    tasks = []
    for subj, action_id, trial, cam in itertools.product(subjects, range(1, 16), trials, cameras):
        tasks.append((subj, actions[action_id - 1], trial, cam,
                      raw_data_root, output_root, frame_skip))

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    with Pool(processes=num_workers) as pool:
        results = pool.map(process_sequence, tasks)

    for r in results:
        print(r)

if __name__ == '__main__':
    FLAGS(sys.argv)
    main(FLAGS.source_dir, FLAGS.out_dir, FLAGS.frame_skip, FLAGS.num_workers)
