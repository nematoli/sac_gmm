# pytorch_sac
#
# Copyright (c) 2019 Denis Yarats
#
# The following code is a derative work from the Denis Yarats,
# which is licensed "MIT License".
#
# Source: https://github.com/denisyarats/pytorch_sac

# from torch.utils.tensorboard import SummaryWriter
import os
import csv
import torch
import wandb
from termcolor import colored
from collections import defaultdict

COMMON_TRAIN_FORMAT = [
    ("episode", "E", "int"),
    ("step", "S", "int"),
    ("episode_reward", "R", "float"),
    ("duration", "D", "time"),
]

COMMON_EVAL_FORMAT = [("episode", "E", "int"), ("step", "S", "int"), ("episode_reward", "R", "float")]


AGENT_TRAIN_FORMAT = {
    "sac_gmm": [
        ("batch_reward", "B_REW", "float"),
        ("actor_loss", "A_LOSS", "float"),
        ("critic_loss", "C_LOSS", "float"),
        ("alpha_loss", "ALP_LOSS", "float"),
        ("alpha_value", "ALP_VAL", "float"),
        ("actor_entropy", "A_ENT", "float"),
        ("ae_loss", "AE_LOSS", "float"),
        ("ae_rec_loss", "REC_LOSS", "float"),
        ("ar_latent_loss", "LAT_LOSS", "float"),
    ]
}


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, file_name, formating):
        self._csv_file_name = self._prepare_file(file_name, "csv")
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._csv_file = open(self._csv_file_name, "w")
        self._csv_writer = None

    def _prepare_file(self, prefix, suffix):
        file_name = f"{prefix}.{suffix}"
        if os.path.exists(file_name):
            os.remove(file_name)
        return file_name

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith("train"):
                key = key[len("train") + 1 :]
            else:
                key = key[len("eval") + 1 :]
            key = key.replace("/", "_")
            data[key] = meter.value()
        return data

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=sorted(data.keys()), restval=0.0)
            self._csv_writer.writeheader()
        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _format(self, key, value, ty):
        if ty == "int":
            value = int(value)
            return f"{key}: {value}"
        elif ty == "float":
            return f"{key}: {value:.04f}"
        elif ty == "time":
            return f"{key}: {value:04.1f} s"
        else:
            raise f"invalid format type: {ty}"

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, "yellow" if prefix == "train" else "green")
        pieces = [f"| {prefix: <14}"]
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print(" | ".join(pieces))

    def dump(self, step, prefix, save=True):
        if len(self._meters) == 0:
            return
        if save:
            data = self._prime_meters()
            data["step"] = step
            self._dump_to_csv(data)
            self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(
        self,
        log_dir,
        save_wb=False,
        cfg=None,
        log_frequency=10000,
        agent="sacgmm",
    ):
        self._log_dir = log_dir
        self._log_frequency = log_frequency

        self.save_wb = save_wb
        if self.save_wb:
            config = {
                "max_steps": cfg.max_steps,
                "num_init_steps": cfg.num_init_steps,
                "eval_frequency": cfg.eval_frequency,
                "num_eval_episodes": cfg.num_eval_episodes,
                "gmm_window_size": cfg.gmm_window_size,
                "env_max_episode_steps": cfg.gmm_max_episode_steps,
                "batch_size": cfg.agent.batch_size,
                "lr_actor": cfg.agent.actor_lr,
                "lr_critic": cfg.agent.critic_lr,
                "lr_alpha": cfg.agent.alpha_lr,
                "lr_ae": cfg.agent.ae_lr,
                "hidden_dim_actor": cfg.agent.actor.hidden_dim,
                "hidden_depth_actor": cfg.agent.actor.hidden_depth,
                "hidden_dim_critic": cfg.agent.critic.hidden_dim,
                "hidden_depth_critic": cfg.agent.critic.hidden_depth,
                "hidden_dim_ae": cfg.agent.autoencoder.hidden_dim,
            }
            wandb.init(project=agent, entity="in-ac", config=config)

        # each agent has specific output format for training
        assert agent in AGENT_TRAIN_FORMAT
        train_format = COMMON_TRAIN_FORMAT + AGENT_TRAIN_FORMAT[agent]
        self._train_mg = MetersGroup(os.path.join(log_dir, "train"), formating=train_format)
        self._eval_mg = MetersGroup(os.path.join(log_dir, "eval"), formating=COMMON_EVAL_FORMAT)

    def log(self, key, value):
        if "video" in key:
            self.log_video(key, value)
            return
        if type(value) == torch.Tensor:
            value = value.item()
        if self.save_wb:
            wb_key = key.replace("/", "_")
            wandb.log({wb_key: value})
        mg = self._train_mg if key.startswith("train") else self._eval_mg
        mg.log(key, value, 1)

    def log_video(self, key, filepath):
        if self.save_wb:
            wb_key = key.split("/")[-1]
            wandb.log({wb_key: wandb.Video(filepath, fps=15, format="gif")})

    def log_params(self, agent, fname=None, actor=False, critic=False):
        self.weights_dir = os.path.join(self._log_dir, "weights")
        os.makedirs(self.weights_dir, exist_ok=True)
        if fname is None:
            fname = ""
        if actor:
            torch.save(
                agent.actor.trunk.state_dict(),
                os.path.join(self.weights_dir, f"actor_{fname}.pth"),
            )
            torch.save(
                agent.critic.Q1.state_dict(),
                os.path.join(self.weights_dir, f"critic_q1_{fname}.pth"),
            )
        if critic:
            torch.save(
                agent.critic.Q2.state_dict(),
                os.path.join(self.weights_dir, f"critic_q2_{fname}.pth"),
            )

    def dump(self, step, save=True, ty=None):
        if ty is None:
            self._train_mg.dump(step, "train", save)
            self._eval_mg.dump(step, "eval", save)
        elif ty == "eval":
            self._eval_mg.dump(step, "eval", save)
        elif ty == "train":
            self._train_mg.dump(step, "train", save)
        else:
            raise f"invalid log type: {ty}"
