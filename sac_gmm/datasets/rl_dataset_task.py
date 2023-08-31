from sac_gmm.datasets.replay_buffer_task import ReplayBufferTask
from torch.utils.data.dataset import IterableDataset
from typing import Iterator


class RLDatasetTask(IterableDataset):
    """
    Iterable Dataset containing the ReplayBuffer
    which will be updated with new experiences during training
    """

    def __init__(self, buffer: ReplayBufferTask, batch_size: int = 64) -> None:
        """
        Args:
            buffer: replay buffer
            sample_size: number of experiences to sample at a time
        """
        self.buffer = buffer
        self.batch_size = batch_size

    def __iter__(self) -> Iterator:
        states, skill_ids, actions, rewards, next_states, next_skill_ids, dones = self.buffer.sample(self.batch_size)
        for i in range(len(dones)):
            yield states[i], skill_ids[i], actions[i], rewards[i], next_states[i], next_skill_ids[i], dones[i]
