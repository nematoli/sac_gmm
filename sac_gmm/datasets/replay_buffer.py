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
        self.unsaved_transitions = 0
        self.curr_file_idx = 1
        self.replay_buffer = deque(maxlen=int(max_capacity))
        self.save_dir = save_dir

    def __len__(self) -> int:
        return len(self.replay_buffer)

    def add(self, state, action, reward, next_state, done):
        """
        This method adds a transition to the replay buffer.
        """
        transition = Transition(state, action, reward, next_state, done)
        self.replay_buffer.append(transition)
        self.unsaved_transitions += 1

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
        if self.unsaved_transitions > 0:
            p = Path(self.save_dir)
            p.mkdir(parents=True, exist_ok=True)
            final_rb_index = len(self.replay_buffer)
            start_rb_index = len(self.replay_buffer) - self.unsaved_transitions
            for replay_buffer_index in range(start_rb_index, final_rb_index):
                transition = self.replay_buffer[replay_buffer_index]
                file_name = "%s/transition_%09d.npz" % (self.save_dir, self.curr_file_idx)
                np.savez(
                    file_name,
                    state=transition.state,
                    action=transition.action,
                    next_state=transition.next_state,
                    reward=transition.reward,
                    done=transition.done,
                )
                self.curr_file_idx += 1
            # Logging
            if self.unsaved_transitions == 1:
                log_rank_0("Saved file with index : %09d" % (self.curr_file_idx - 1))
            else:
                log_rank_0(
                    "Saved files with indices : %09d - %09d"
                    % (
                        self.curr_file_idx - self.unsaved_transitions,
                        self.curr_file_idx - 1,
                    )
                )
            self.unsaved_transitions = 0
            return True
        return False
