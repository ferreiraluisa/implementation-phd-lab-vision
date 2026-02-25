import random
from collections import defaultdict
from torch.utils.data import Sampler

class MixedShardBatchSampler(Sampler):
    """
    Shuffle + locality, but mix shards inside each batch:
      - keep per-shard buckets (like your current sampler)
      - pick K shards at a time
      - draw batch_size/K samples from each (round-robin)
    """
    def __init__(self, dataset, batch_size, shards_per_batch=4,
                 shuffle=True, drop_last=True, seed=0):
        assert batch_size % shards_per_batch == 0
        self.dataset = dataset
        self.batch_size = batch_size
        self.K = shards_per_batch
        self.per_shard = batch_size // shards_per_batch
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        buckets = defaultdict(list)
        for idx, (clip, var) in enumerate(dataset._items):
            buckets[clip["shard_id"]].append(idx)
        self.buckets = {k: v for k, v in buckets.items()}

    def set_epoch(self, epoch):
        self.seed = epoch

    def __iter__(self):
        rng = random.Random(self.seed)
        shard_ids = list(self.buckets.keys())
        if self.shuffle:
            rng.shuffle(shard_ids)

        # make per-shard streams
        streams = {}
        for sid in shard_ids:
            inds = self.buckets[sid].copy()
            if self.shuffle:
                rng.shuffle(inds)
            streams[sid] = inds

        # active shards with remaining samples
        active = [sid for sid in shard_ids if len(streams[sid]) > 0]

        while len(active) >= self.K:
            chosen = rng.sample(active, self.K) if self.shuffle else active[:self.K]
            batch = []
            for sid in chosen:
                take = min(self.per_shard, len(streams[sid]))
                batch.extend(streams[sid][:take])
                del streams[sid][:take]
                if len(streams[sid]) == 0:
                    active.remove(sid)

            if len(batch) < self.batch_size:
                if self.drop_last:
                    continue
                # if not drop_last, yield partial
            yield batch

    def __len__(self):
        # rough lower bound; exact len depends on shard distributions
        total = len(self.dataset)
        return total // self.batch_size if self.drop_last else (total + self.batch_size - 1) // self.batch_size