from sac_gmm.datasets.replay_buffer_episode import ReplayBufferEpisode, SeqSampler
from torch.utils.data.dataset import IterableDataset
from typing import Iterator
import pickle
import gzip


class CALVINPlayDataset(IterableDataset):
    """
    Iterable Dataset containing the ReplayBuffer
    which will be updated with new experiences during training
    """

    def __init__(
        self, horizon, data_path, train=True, train_split: float = 0.99, batch_size: int = 64, precision: int = 32
    ) -> None:
        """
        Args:
            buffer: replay buffer
            sample_size: number of experiences to sample at a time
        """
        self.batch_size = batch_size
        buffer_keys = ["ob", "ac", "done"]
        sampler = SeqSampler(horizon + 1)

        self._pretrain_buffer = ReplayBufferEpisode(buffer_keys, None, sampler.sample_func_tensor, precision)
        data = pickle.load(gzip.open(data_path, "rb"))
        data_size = len(data)

        for i, d in enumerate(data):
            if (train and i < data_size * train_split) or (not train and i >= data_size * train_split):
                if len(d["obs"]) < len(d["dones"]):
                    continue  # Skip incomplete trajectories.
                d["obs"] = d["obs"][:, :21]
                new_d = dict(ob=d["obs"], ac=d["actions"], done=d["dones"])
                new_d["done"][-1] = 1.0  # Force last step to be done.
                self._pretrain_buffer.store_episode(new_d, False)

    def __iter__(self) -> Iterator:
        batch = self._pretrain_buffer.sample(self.batch_size)
        yield batch
