import random
from collections import defaultdict
from torch.utils.data import Sampler

""""
Resolves data loading taking so much when use shards to make many variants per clip (n_vars=10) and many clips per shard (e.g. 100), which causes a lot of thrashing when the DataLoader workers load shards in random order and then only use a few samples from each shard before moving on to the next one.
"""
class ShardGroupedBatchSampler(Sampler):
    """
    Shuffle with locality:
      - shuffle order of shards
      - shuffle indices inside each shard
      - yield batches mostly from same shard (prevents shard thrash)
    """
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        buckets = defaultdict(list)
        # dataset._items = [(clip_dict, var_offset), ...]
        for idx, (clip, var) in enumerate(dataset._items):
            buckets[clip["shard_id"]].append(idx)
        self.buckets = dict(buckets)

    def __iter__(self):
        rng = random.Random(self.seed)
        shard_ids = list(self.buckets.keys())
        if self.shuffle:
            rng.shuffle(shard_ids)

        for sid in shard_ids:
            inds = self.buckets[sid]
            if self.shuffle:
                rng.shuffle(inds)

            for i in range(0, len(inds), self.batch_size):
                batch = inds[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch

    def __len__(self):
        n = 0
        for inds in self.buckets.values():
            if self.drop_last:
                n += len(inds) // self.batch_size
            else:
                n += (len(inds) + self.batch_size - 1) // self.batch_size
        return n