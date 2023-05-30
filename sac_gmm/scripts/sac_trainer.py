import hydra
import os
import sys
from pathlib import Path
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

cwd_path = Path(__file__).absolute().parents[0]  # scripts
sac_gmm = cwd_path.parents[0]
root = sac_gmm.parents[0]
sys.path.insert(0, sac_gmm.as_posix())
sys.path.insert(0, root.as_posix())


class SACTrainer(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg

        self.model_dir = Path(self.cfg.exp_dir) / "model_weights/"
        os.makedirs(self.model_dir, exist_ok=True)

    def run(self):
        date_time = "_".join(self.cfg.exp_dir.split("/")[-2:])
        save_filename = date_time

        # Model Checkpointing
        model_dir = Path(self.model_dir) if self.model_dir is not None else None

        checkpoint_val_callback = ModelCheckpoint(
            monitor="eval_episode_return",
            dirpath=model_dir,
            filename="%s_{step:06d}_{val_episode_return:.3f}" % save_filename,
            save_top_k=10,
            verbose=True,
            mode="max",
            save_last=True,
        )

        # Wandb Logger
        name = "%s_%s" % (self.cfg.wandb.developer, save_filename)
        name = "%s_%s" % (name, self.cfg.wandb.comment) if "comment" in self.cfg.wandb else name
        wandb_logger = WandbLogger(
            entity=self.cfg.wandb.entity,
            project=self.cfg.wandb.project,
            name=name,
            resume="allow",
        )
        hparams = {
            "num_train_steps": self.cfg.num_train_steps,
            "num_seed_steps": self.cfg.num_seed_steps,
            "gmm_window_size": self.cfg.gmm_window_size,
            "max_episode_steps": self.cfg.max_episode_steps,
            "batch_size": self.cfg.rl.batch_size,
            "lr_actor": self.cfg.rl.actor_lr,
            "lr_critic": self.cfg.rl.critic_lr,
            "lr_alpha": self.cfg.rl.alpha_lr,
            "lr_ae": self.cfg.rl.ae_lr,
            "compact_rep_size": self.cfg.rl.compact_rep_size,
            "hidden_dim_actor": self.cfg.rl.actor.hidden_dim,
            "hidden_depth_actor": self.cfg.rl.actor.hidden_depth,
            "hidden_dim_critic": self.cfg.rl.critic.hidden_dim,
            "hidden_depth_critic": self.cfg.rl.critic.hidden_depth,
            "late_fusion": self.cfg.rl.late_fusion,
            "shared_encoder": self.cfg.rl.shared_encoder,
            "fixed_encoder": self.cfg.rl.fixed_encoder,
        }
        wandb_logger.experiment.config.update(hparams)

        # Lightning Trainer
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            num_nodes=1,
            max_steps=self.cfg.num_train_steps,
            callbacks=[checkpoint_val_callback],
            logger=wandb_logger,
        )

        if model_dir is None:
            ckpt_path = None
        else:
            ckpt_path = model_dir / "last.ckpt"
            ckpt_path = ckpt_path if ckpt_path.is_file() else None

        # Algorithm - SAC or SACGMM etc
        model = hydra.utils.instantiate(self.cfg.rl)

        trainer.fit(model, ckpt_path=ckpt_path)


@hydra.main(version_base="1.1", config_path="../../config", config_name="sac_trainer")
def main(cfg: DictConfig) -> None:
    hydra_out_dir = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    cfg.exp_dir = hydra_out_dir

    pl_trainer = SACTrainer(cfg)
    pl_trainer.run()


if __name__ == "__main__":
    main()
