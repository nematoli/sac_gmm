import logging
from pathlib import Path
import numpy as np
from collections import deque, namedtuple
from tqdm import tqdm
from pytorch_lightning.utilities import rank_zero_only

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    def __init__(self, save_dir, max_capacity=5e6):
        self.last_saved_idx = 0
        self.replay_buffer = deque(maxlen=int(max_capacity))
        self.save_dir = save_dir
        self.load()

    def __len__(self) -> int:
        return len(self.replay_buffer)

    def add(self, state, action, reward, next_state, done):
        """
        This method adds a transition to the replay buffer.
        """
        transition = Transition(state, action, reward, next_state, done)
        self.replay_buffer.append(transition)

    def sample(self, batch_size: int):
        indices = np.random.choice(
            len(self.replay_buffer),
            min(len(self.replay_buffer), batch_size),
            replace=False,
        )
        states, actions, rewards, next_states, dones = zip(*[self.replay_buffer[idx] for idx in indices])
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=bool),
        )

    def save(self):
        if self.save_dir is None:
            return False
        p = Path(self.save_dir)
        p.mkdir(parents=True, exist_ok=True)
        num_entries = len(self.replay_buffer)
        for i in range(self.last_saved_idx, num_entries):
            transition = self.replay_buffer[i]
            file_name = "%s/transition_%d.npz" % (self.save_dir, i)
            np.savez(
                file_name,
                state=transition.state,
                action=transition.action,
                reward=transition.reward,
                next_state=transition.next_state,
                done=transition.done,
            )
        log_rank_0("Saved transitions with indices : %d - %d" % (self.last_saved_idx, i))
        self.last_saved_idx = i

    def load(self):
        if self.save_dir is None:
            return False
        p = Path(self.save_dir)
        if p.is_dir():
            p = p.glob("*.npz")
            files = [x for x in p if x.is_file()]
            if len(files) > 0:
                for file in files:
                    data = np.load(file, allow_pickle=True)
                    transition = Transition(
                        data["state"].item(),
                        data["action"],
                        data["reward"].item(),
                        data["next_state"].item(),
                        data["done"].item(),
                    )
                    self.replay_buffer.append(transition)
                log_rank_0("Replay buffer loaded successfully")
            else:
                log_rank_0("No files were found in path %s" % (self.save_dir))
        else:
            log_rank_0("Path %s does not have an appropiate directory address" % (self.save_dir))
