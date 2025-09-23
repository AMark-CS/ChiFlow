"""Pytorch script for training FoldFlow.

Enhancements:
 - Added real-time logging of designability_fraction and diversity_fraction to Weights & Biases.
     * designability_fraction: fraction of evaluated samples whose tm_score >= designability_tm_thresh.
     * diversity_fraction: fraction of pairwise CA-RMSD distances (between generated samples) >= diversity_rmsd_thresh.
         (Computed on CA atoms; O(N^2); capped by diversity_max_samples or lightweight subset size.)
 - Full evaluation steps compute fractions from the full eval set.
 - Between full evals, a lightweight fraction evaluation (configurable) runs periodically on a
     small validation subset so curves update in near real-time.
Config additions (add under experiment in YAML):
    designability_tm_thresh: 0.5        # TM-score threshold used for designability_fraction
    diversity_rmsd_thresh: 5.0          # CA RMSD threshold for diversity_fraction
    diversity_max_samples: 40           # Max samples for diversity pairwise RMSD (full eval)
    fraction_eval_freq: 200             # Lightweight fraction eval frequency (steps); 0 disables real-time eval
    fraction_eval_num_samples: 8        # Number of validation samples used in lightweight fraction eval
Notes:
    - If fraction_eval_freq divides the global step and no full evaluation occurs, the lightweight
      evaluation runs and updates cached fraction metrics.
    - Cached fraction metrics are logged every step (if available) so wandb curves are continuous.
"""
import os
import torch.multiprocessing as mp
import torch  # Added missing torch import

# Set the multiprocessing start method to 'spawn' to resolve CUDA initialization issues in forked subprocesses.
mp.set_start_method('spawn', force=True)

# This line magically changes some tensors to double precision
# so we need to reset the default dtype later.
os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import copy
import logging
import time
from functools import lru_cache
from collections import defaultdict, deque

import GPUtil
import hydra
import numpy as np
import pandas as pd
import torch
import tree
import wandb
import mdtraj as md  # used for diversity RMSD calculations
from einops import rearrange
import foldflow.utils.experiments_utils as eu
from hydra.core.hydra_config import HydraConfig
from lightning import Fabric
from omegaconf import DictConfig, OmegaConf
from torch.nn import DataParallel as DP
from foldflow.utils.so3_helpers import hat_inv, pt_to_identity

from foldflow.data import all_atom, pdb_data_loader
from foldflow.data import utils as du
from foldflow.models import se3_fm
from foldflow.models.components import network
from openfold.utils import rigid_utils as ru
from tools.analysis import metrics
from tools.analysis import utils as au


class Experiment:
    def __init__(
        self,
        *,
        conf: DictConfig,
        model = None,
    ):
        """Initialize experiment.

        Args:
            exp_cfg: Experiment configuration.
        """
        self.first_batch = None
        self._log = logging.getLogger(__name__)
        self._available_gpus = "".join(
            [str(x) for x in GPUtil.getAvailable(order="memory", limit=8)]
        )

        # Configs
        self._conf = conf
        self._exp_conf = conf.experiment
        if HydraConfig.initialized() and "num" in HydraConfig.get().job:
            self._exp_conf.name = f"{self._exp_conf.name}_{HydraConfig.get().job.num}"
        self._fm_conf = conf.flow_matcher
        self._model_conf = conf.model
        self._data_conf = conf.data
        self._wandb_conf = conf.wandb
        self._use_wandb = self._wandb_conf.use_wandb
        self._use_ddp = self._exp_conf.use_ddp
        # 1. initialize ddp info if in ddp mode
        # 2. silent rest of logger when use ddp mode
        # 3. silent wandb logger
        # 4. unset checkpoint path if rank is not 0 to avoid saving checkpoints and evaluation
        print(f"Number of threads {self._exp_conf.torch_num_threads}")
        torch.set_num_threads(self._exp_conf.torch_num_threads)
        # reduce matmul precision for better performance on GPU
        torch.set_float32_matmul_precision("medium")
        torch.set_default_dtype(torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = True
        self._master_proc = True

        if self._use_ddp:
            from lightning.fabric.strategies import DDPStrategy

            strategy = DDPStrategy(find_unused_parameters=False)
            self.fabric = Fabric(
                accelerator="cuda", devices=self._exp_conf.num_gpus, strategy=strategy
            )
            self.fabric.launch()

            torch.backends.cuda.matmul.allow_tf32 = True
            self._log.info(f"Using DDP with {self.fabric.global_rank} rank")
            print(
                f"Torch uses cuddn {torch.backends.cudnn.enabled} and"
                f" cudnn benchmark {torch.backends.cudnn.benchmark}"
            )

            self.ddp_info = eu.get_ddp_info()
            self._master_proc = self.fabric.global_rank == 0
            self._global_rank = self.fabric.global_rank
            print(
                f"RANK: {self.fabric.global_rank} | master process: {self._master_proc}"
            )

            if self.fabric.global_rank != 0:
                self._log.addHandler(logging.NullHandler())
                self._log.setLevel("ERROR")
                self._use_wandb = False
                # self._exp_conf.full_ckpt_dir = None

        ckpt_model, ckpt_opt = self.handle_warmstart(conf)

        if self._use_ddp and self.fabric.global_rank != 0:
            self._exp_conf.full_ckpt_dir = None

        # Initialize experiment objects
        if self._model_conf.model_name == "chiflow":
            # For ChiFlow, use ChiFlowMatcher instead of SE3FlowMatcher
            from foldflow.models.chiflow import ChiFlowMatcher
            self._flow_matcher = ChiFlowMatcher(self._model_conf)
        else:
            # For SE3 models, use SE3FlowMatcher
            self._flow_matcher = se3_fm.SE3FlowMatcher(self._fm_conf)

        # Thresholds for new fraction metrics (allow missing in older configs)
        self._designability_tm_thresh = getattr(self._exp_conf, "designability_tm_thresh", 0.5)
        self._diversity_rmsd_thresh = getattr(self._exp_conf, "diversity_rmsd_thresh", 5.0)
        self._diversity_max_samples = getattr(self._exp_conf, "diversity_max_samples", 40)
        # Lightweight (real-time) fraction evaluation settings (default every 200 steps)
        self._fraction_eval_freq = getattr(self._exp_conf, "fraction_eval_freq", 200)
        self._fraction_eval_num_samples = getattr(self._exp_conf, "fraction_eval_num_samples", 8)
        # Cached latest fractions for logging each step (even when not recomputed)
        self._last_designability_fraction = None
        self._last_diversity_fraction = None

        self._model = model
        if self._model is None:
            if self._model_conf.model_name == "chiflow":
                # ChiFlow uses its own model architecture
                from foldflow.models.chiflow import ChiFlowModel
                self._model = ChiFlowModel(self._model_conf)
            else:
                # SE3 models use VectorFieldNetwork
                self._model = network.VectorFieldNetwork(self._model_conf, self.flow_matcher)
            if ckpt_model is not None:
                ckpt_model = {k.replace("module.", ""): v for k, v in ckpt_model.items()}
                ckpt_model = {
                    k.replace("score_model.", "vectorfield."): v
                    for k, v in ckpt_model.items()
                }
                self._model.load_state_dict(ckpt_model, strict=True)

            num_parameters = sum(p.numel() for p in self._model.parameters())
            self._exp_conf.num_parameters = num_parameters
            self._log.info(f"Number of model parameters {num_parameters}")
            self._optimizer = torch.optim.Adam(
                self._model.parameters(), lr=self._exp_conf.learning_rate
            )
            if ckpt_opt is not None:
                self._optimizer.load_state_dict(ckpt_opt)
                if conf.experiment.use_gpu:
                    for state in self._optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda()

        if self._exp_conf.full_ckpt_dir is not None:
            # Set-up checkpoint location
            # Original config gives something like ./ckpt/<name>/
            base_ckpt_dir = self._exp_conf.full_ckpt_dir
            # Ensure base dir exists
            os.makedirs(base_ckpt_dir, exist_ok=True)
            # Per-run timestamped subdirectory
            run_start_time = time.strftime("%Y%m%d_%H%M%S")
            run_ckpt_dir = os.path.join(base_ckpt_dir, run_start_time)
            os.makedirs(run_ckpt_dir, exist_ok=True)
            # Update config to point to run-specific directory
            self._exp_conf.full_ckpt_dir = run_ckpt_dir
            self._log.info(f"Checkpoints saved to per-run directory: {run_ckpt_dir}")
            # Create/refresh a symlink 'latest' inside base_ckpt_dir pointing to this run for convenience
            latest_link = os.path.join(base_ckpt_dir, 'latest')
            try:
                if os.path.islink(latest_link) or os.path.exists(latest_link):
                    if os.path.islink(latest_link):
                        os.unlink(latest_link)
                    else:
                        # If a normal dir/file named latest exists, skip to avoid accidental deletion
                        self._log.warning(f"Cannot create latest symlink, path exists and is not a symlink: {latest_link}")
                if not os.path.exists(latest_link):
                    os.symlink(run_ckpt_dir, latest_link)
            except OSError as e:
                self._log.warning(f"Failed to create latest symlink: {e}")
        else:
            self._log.info("Checkpoint not being saved.")
        if self._exp_conf.eval_dir is not None:
            # Mirror the timestamped run directory name for eval outputs for easier association
            # Reuse the same run_start_time if we created one above; otherwise create a new timestamp
            if 'run_ckpt_dir' in locals():
                run_timestamp = os.path.basename(run_ckpt_dir)
            else:
                run_timestamp = time.strftime("%Y%m%d_%H%M%S")
            eval_dir = os.path.join(self._exp_conf.eval_dir, self._exp_conf.name, run_timestamp)
            self._exp_conf.eval_dir = eval_dir
            os.makedirs(eval_dir, exist_ok=True)
            self._log.info(f"Evaluation saved to: {eval_dir}")
        else:
            self._exp_conf.eval_dir = os.devnull
            self._log.info("Evaluation will not be saved.")
        self._aux_data_history = deque(maxlen=100)

        # DEBUG Variables
        self._first_train_feats = None
        self._global_rank = 0

    def handle_warmstart(self, conf):
        # Warm starting
        ckpt_model = None
        ckpt_opt = None
        self.trained_epochs = 0
        self.trained_steps = 0
        if not conf.experiment.warm_start:
            return None, None

        assert conf.experiment.warm_start in ["auto", "force"]

        # check path exists
        full_ckpt_dir = conf.experiment.full_ckpt_dir
        print(f"THIS IS TH FULL CONF!\n {conf.experiment}")
        if full_ckpt_dir is not None and not os.path.exists(full_ckpt_dir):
            if conf.experiment.warm_start == "auto":
                return None, None
            if conf.experiment.warm_start == "force":
                raise ValueError(f"full_ckpt_dir {full_ckpt_dir} does not exist")

        ckpt_files = [x for x in os.listdir(full_ckpt_dir) if "pkl" in x or ".pth" in x]
        if len(ckpt_files) == 0:
            if conf.experiment.warm_start == "auto":
                return None, None
            if conf.experiment.warm_start == "force":
                raise ValueError(f"full_ckpt_dir {full_ckpt_dir} has no checkpoints")

        self._log.info(f"Warm starting from: {full_ckpt_dir}")

        ckpt_name = ckpt_files[0]
        if len(ckpt_files) != 1:
            paths = [os.path.join(full_ckpt_dir, ckpt_file) for ckpt_file in ckpt_files]
            ckpt_name = max(paths, key=os.path.getmtime).split("/")[-1]
            self._log.info("Loading most recent ckpt")
        ckpt_path = os.path.join(full_ckpt_dir, ckpt_name)
        self._log.info(f"Loading checkpoint from {ckpt_path}")
        ckpt_pkl = du.read_pkl(ckpt_path, use_torch=True)
        ckpt_model = ckpt_pkl["model"]

        if conf.experiment.use_warm_start_conf:
            OmegaConf.set_struct(conf, False)
            conf = OmegaConf.merge(conf, ckpt_pkl["conf"])
            OmegaConf.set_struct(conf, True)
        conf.experiment.warm_start = full_ckpt_dir

        # For compatibility with older checkpoints.
        if "optimizer" in ckpt_pkl:
            ckpt_opt = ckpt_pkl["optimizer"]
        if "epoch" in ckpt_pkl:
            self.trained_epochs = ckpt_pkl["epoch"]
        if "step" in ckpt_pkl:
            self.trained_steps = ckpt_pkl["step"]
        return ckpt_model, ckpt_opt

    @property
    def flow_matcher(self):
        return self._flow_matcher

    @property
    def model(self):
        return self._model

    @property
    def conf(self):
        return self._conf

    def create_dataset(self):
        train_dataset = pdb_data_loader.PdbDataset(
            data_conf=self._data_conf,
            gen_model=self._flow_matcher,
            is_training=True,
            is_OT=self._fm_conf.ot_plan,
            ot_fn=self._fm_conf.ot_fn,
            reg=self._fm_conf.reg,
        )

        valid_dataset = pdb_data_loader.PdbDataset(
            data_conf=self._data_conf,
            gen_model=self._flow_matcher,
            is_OT=self._fm_conf.ot_plan,
            ot_fn=self._fm_conf.ot_fn,
            reg=self._fm_conf.reg,
            is_training=False,
        )
        if self._use_ddp:
            train_sampler = pdb_data_loader.DistributedTrainSampler(
                data_conf=self._data_conf,
                dataset=train_dataset,
                batch_size=self._exp_conf.batch_size,
                sample_mode=self._exp_conf.sample_mode,
                rank=self.fabric.global_rank,
                max_squared_res=self._exp_conf.max_squared_res,
                num_gpus=self._exp_conf.num_gpus,  # TODO fix arg based on actual fabric
            )
        else:
            train_sampler = pdb_data_loader.TrainSampler(
                data_conf=self._data_conf,
                dataset=train_dataset,
                batch_size=self._exp_conf.batch_size,
                sample_mode=self._exp_conf.sample_mode,
                max_squared_res=self._exp_conf.max_squared_res,
                num_gpus=self._exp_conf.num_gpus,
            )
        valid_sampler = None
        num_workers = self._exp_conf.num_loader_workers

        train_loader = du.create_data_loader(
            train_dataset,
            sampler=train_sampler,
            np_collate=False,
            length_batch=True,
            batch_size=self._exp_conf.batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            max_squared_res=self._exp_conf.max_squared_res,
        )

        valid_loader = du.create_data_loader(
            valid_dataset,
            sampler=valid_sampler,
            np_collate=False,
            length_batch=True,
            batch_size=self._exp_conf.eval_batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        if self._exp_conf.use_ddp:
            train_loader = self.fabric.setup_dataloaders(
                train_loader, use_distributed_sampler=False
            )
        return train_loader, valid_loader, train_sampler, valid_sampler

    def init_wandb(self):
        self._log.info("Initializing Wandb.")
        conf_dict = OmegaConf.to_container(self._conf, resolve=True)
        if self._exp_conf.run_id is None:
            self._exp_conf.run_id = wandb.util.generate_id()
        wandb.init(
            project=self._wandb_conf.project,
            entity=self._wandb_conf.entity,
            name=self._exp_conf.name,
            config=dict(eu.flatten_dict(conf_dict)),
            dir=self._wandb_conf.dir,
            id=self._exp_conf.run_id,
            tags=self._wandb_conf.tags,
            group=self._wandb_conf.group,
            mode="offline" if self._wandb_conf.offline else "online",
            job_type=self._wandb_conf.job_type,
            resume="auto" if self._exp_conf.warm_start is not None else None,
        )
        self._wandb_conf.dir = wandb.run.dir
        self._log.info(
            f"Wandb: run_id={self._exp_conf.run_id}, run_dir={self._wandb_conf.dir}"
        )
        # Define fraction metrics after Wandb init
        wandb.define_metric("designability_fraction", step_metric="_step")
        wandb.define_metric("diversity_fraction", step_metric="_step")

    def start_training(self, return_logs=False):
        # print(f"Start-training-"*10)
        # Set environment variables for which GPUs to use.
        if HydraConfig.initialized() and "num" in HydraConfig.get().job:
            replica_id = int(HydraConfig.get().job.num)
        else:
            replica_id = 0
        if self._use_wandb and replica_id == 0:
            self.init_wandb()
        assert not self._exp_conf.use_ddp or self._exp_conf.use_gpu

        # GPU mode
        if torch.cuda.is_available() and self._exp_conf.use_gpu:
            # single GPU mode
            if self._exp_conf.num_gpus == 1:
                try:
                    gpu_id = self._available_gpus[replica_id]
                    device = f"cuda:{gpu_id}"
                except IndexError:
                    device = "cuda:0"
                    self._log.warning("Error on available gpus, trying with device 0")
                self._model = self.model.to(device)
                # Ensure all model parameters are on the correct device
                self._model = self._model.to(device)
                # Also move flow matcher to the correct device
                if hasattr(self, '_flow_matcher') and self._flow_matcher is not None:
                    self._flow_matcher = self._flow_matcher.to(device)
                self._log.info(f"Using device: {device}")
                self._log.info(f"Model device: {next(self._model.parameters()).device}")
                if hasattr(self, '_flow_matcher') and self._flow_matcher is not None:
                    self._log.info(f"Flow matcher device: {next(self._flow_matcher.parameters()).device}")
            # muti gpu mode
            elif self._exp_conf.num_gpus > 1:
                # DDP mode
                if self._use_ddp:
                    self._model, self._optimizer = self.fabric.setup(
                        self._model, self._optimizer
                    )
                    device = self.fabric.device
                    # Move flow matcher to device in DDP mode
                    if hasattr(self, '_flow_matcher') and self._flow_matcher is not None:
                        self._flow_matcher = self.fabric.setup_module(self._flow_matcher)
                    self._log.info(f"Using device: {device}")
                # DP mode
                else:
                    device_ids = [
                        f"cuda:{i}"
                        for i in self._available_gpus[: self._exp_conf.num_gpus]
                    ]
                    if len(self._available_gpus) > self._exp_conf.num_gpus:
                        raise ValueError(
                            f"require {self._exp_conf.num_gpus} GPUs, but only {len(self._available_gpus)} GPUs available "
                        )
                    self._log.info(
                        f"Multi-GPU training on GPUs in DP mode: {device_ids}"
                    )
                    gpu_id = self._available_gpus[replica_id]
                    device = f"cuda:{gpu_id}"
                    self._model = DP(self._model, device_ids=device_ids)
                    self._model = self.model.to(device)
        else:
            device = "cpu"
            self._log.info(f"Using device: {device}")
            self._model = self.model.to(device)
            # Move flow matcher to CPU
            if hasattr(self, '_flow_matcher') and self._flow_matcher is not None:
                self._flow_matcher = self._flow_matcher.to(device)

        self._model.train()
        (
            train_loader,
            valid_loader,
            train_sampler,
            valid_sampler,
        ) = self.create_dataset()

        logs = []
        for epoch in range(self.trained_epochs, self._exp_conf.num_epoch):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if valid_sampler is not None:
                valid_sampler.set_epoch(epoch)
                # Memory / batch diagnostics (first few steps and periodic)
                if (
                    torch.cuda.is_available()
                    and self.trained_steps <= 5
                ) or (self.trained_steps % 500 == 0 and torch.cuda.is_available()):
                    try:
                        alloc = torch.cuda.memory_allocated() / 1024 ** 2
                        reserved = torch.cuda.memory_reserved() / 1024 ** 2
                        lengths = batch.get('res_mask', torch.ones(1,1)).sum(dim=-1)
                        eff_tokens = lengths.sum().item()
                        self._log.info(
                            f"[Diag] step={self.trained_steps} batch_size={len(lengths)} mean_len={lengths.float().mean():.1f} total_tokens={eff_tokens} mem_alloc={alloc:.1f}MB mem_reserved={reserved:.1f}MB"
                        )
                    except Exception:
                        pass
            self.trained_epochs = epoch
            epoch_log = self.train_epoch(
                train_loader, valid_loader, device, return_logs=return_logs
            )
            if return_logs:
                logs.append(epoch_log)

        self._log.info("Done")
        if return_logs:
            return logs
        return 0

    def update_fn(self, data, debug=False):
        """Updates the state using some data and returns metrics."""
        self._optimizer.zero_grad()
        # torch.autograd.set_detect_anomaly(True, check_nan=True)

        loss, aux_data = self.loss_fn(data)
        if self._use_ddp:
            self.fabric.backward(loss)
        else:
            loss.backward()

        if debug:
            for name, param in self._model.named_parameters():
                if param.grad is None:
                    print(f"NO GRAD FOR PARAMETERS  {name}")

        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
        self._optimizer.step()
        return loss, aux_data

    def train_epoch(self, train_loader, valid_loader, device, return_logs=False):
        log_lossses = defaultdict(list)
        global_logs = []
        log_time = time.time()
        step_time = time.time()

        for train_feats in train_loader:
            if "dummy_batch" in train_feats:
                self._log.error("Dummy batch")
                continue

            if not self._use_ddp:
                def move_to_device(x):
                    if torch.is_tensor(x):
                        return x.to(device)
                    elif isinstance(x, dict):
                        return {k: move_to_device(v) for k, v in x.items()}
                    elif isinstance(x, (list, tuple)):
                        return type(x)(move_to_device(item) for item in x)
                    else:
                        return x
                train_feats = tree.map_structure(move_to_device, train_feats)

            loss, aux_data = self.update_fn(train_feats)

            if return_logs:
                global_logs.append(loss)
            for k, v in aux_data.items():
                log_lossses[k].append(du.move_to_np(v))
            self.trained_steps += 1

            # Logging to terminal
            if (
                self.trained_steps == 1
                or self.trained_steps % self._exp_conf.log_freq == 0
            ):
                elapsed_time = time.time() - log_time
                log_time = time.time()
                step_per_sec = self._exp_conf.log_freq / elapsed_time
                rolling_losses = tree.map_structure(np.mean, log_lossses)
                loss_log = " ".join(
                    [
                        f"{k}={v[0]:.4f}"
                        for k, v in rolling_losses.items()
                        if "batch" not in k
                    ]
                )
                self._log.info(
                    f"[{self.trained_steps}]: {loss_log}, steps/sec={step_per_sec:.5f}"
                )
                log_lossses = defaultdict(list)
            # Take checkpoint
            if ((self.trained_steps % self._exp_conf.ckpt_freq) == 0) or (
                self._exp_conf.early_ckpt and self.trained_steps == 10
            ):
                if self._master_proc and self._exp_conf.full_ckpt_dir is not None:
                    self._log.info("Take checkpoint")
                    ckpt_path = os.path.join(
                        self._exp_conf.full_ckpt_dir, f"step_{self.trained_steps}.pth"
                    )
                    du.write_checkpoint(
                        ckpt_path,
                        self.model.state_dict(),
                        self._conf,
                        self._optimizer.state_dict(),
                        self.trained_epochs,
                        self.trained_steps,
                        logger=self._log,
                        use_torch=True,
                    )
            ckpt_metrics = None
            eval_time = None
            if ((self.trained_steps % self._exp_conf.eval_freq) == 0) or (
                self._exp_conf.early_ckpt and self.trained_steps == 10
            ):
                if self._master_proc:
                    # Run evaluation
                    start_time = time.time()
                    eval_dir = os.path.join(
                        self._exp_conf.eval_dir, f"step_{self.trained_steps}"
                    )
                    os.makedirs(eval_dir, exist_ok=True)

                    self._log.info(
                        f"Running evaluation at EP "
                        f"{self.trained_epochs} step {self.trained_steps} in {eval_dir}"
                    )

                    ckpt_metrics = self.eval_fn(
                        eval_dir,
                        valid_loader,
                        device,
                        noise_scale=self._exp_conf.noise_scale,
                    )
                    eval_time = time.time() - start_time
                    self._log.info(f"Finished evaluation in {eval_time:.2f}s")

            # Remote log to Wandb.
            if self._use_wandb and self._master_proc:
                step_time = time.time() - step_time
                example_per_sec = self._exp_conf.batch_size / step_time
                step_time = time.time()

                # Lightweight fraction evaluation (only if no full eval this step)
                if (
                    self._fraction_eval_freq > 0
                    and self.trained_steps % self._fraction_eval_freq == 0
                    and ckpt_metrics is None  # skip if full eval already computed fractions
                ):
                    self._lightweight_fraction_eval(valid_loader, device)

                if self._model_conf.model_name == "chiflow":
                    # ChiFlow specific logging
                    wandb_logs = {
                        "loss": loss,
                        "dihedral_flow_loss": aux_data["dihedral_flow_loss"],
                        "backbone_loss": aux_data["backbone_loss"],
                        "mirror_constraint_loss": aux_data["mirror_constraint_loss"],
                        "batch_size": aux_data["examples_per_step"],
                        "res_length": aux_data["res_length"],
                        "examples_per_sec": example_per_sec,
                        "num_epochs": self.trained_epochs,
                    }
                else:
                    # SE3 flow matching logging
                    wandb_logs = {
                        "loss": loss,
                        "rotation_loss": aux_data["rot_loss"],
                        "translation_loss": aux_data["trans_loss"],
                        "bb_atom_loss": aux_data["bb_atom_loss"],
                        "dist_mat_loss": aux_data["batch_dist_mat_loss"],
                        "batch_size": aux_data["examples_per_step"],
                        "res_length": aux_data["res_length"],
                        "examples_per_sec": example_per_sec,
                        "num_epochs": self.trained_epochs,
                    }

                    # Stratified losses for SE3
                    wandb_logs.update(
                        eu.t_stratified_loss(
                            du.move_to_np(train_feats["t"]),
                            du.move_to_np(aux_data["batch_rot_loss"]),
                            loss_name="rot_loss",
                        )
                    )

                    wandb_logs.update(
                        eu.t_stratified_loss(
                            du.move_to_np(train_feats["t"]),
                            du.move_to_np(aux_data["batch_trans_loss"]),
                            loss_name="trans_loss",
                        )
                    )

                    wandb_logs.update(
                        eu.t_stratified_loss(
                            du.move_to_np(train_feats["t"]),
                            du.move_to_np(aux_data["batch_bb_atom_loss"]),
                            loss_name="bb_atom_loss",
                        )
                    )

                    wandb_logs.update(
                        eu.t_stratified_loss(
                            du.move_to_np(train_feats["t"]),
                            du.move_to_np(aux_data["batch_dist_mat_loss"]),
                            loss_name="dist_mat_loss",
                        )
                    )

                if ckpt_metrics is not None:
                    wandb_logs["eval_time"] = eval_time
                    for metric_name in metrics.ALL_METRICS:
                        wandb_logs[metric_name] = ckpt_metrics[metric_name].mean()
                    eval_table = wandb.Table(
                        columns=ckpt_metrics.columns.to_list() + ["structure"]
                    )
                    for _, row in ckpt_metrics.iterrows():
                        pdb_path = row["sample_path"]
                        row_metrics = row.to_list() + [wandb.Molecule(pdb_path)]
                        eval_table.add_data(*row_metrics)
                    wandb_logs["sample_metrics"] = eval_table

                    # --- Designability Fraction ---
                    if "tm_score" in ckpt_metrics.columns:
                        designability_fraction = (ckpt_metrics["tm_score"] >= self._designability_tm_thresh).mean()
                        wandb_logs["designability_fraction"] = designability_fraction

                    # --- Diversity Fraction ---
                    # Compute fraction of pairwise CA RMSDs >= threshold.
                    # Limit number of samples for O(N^2) computation.
                    sample_paths = ckpt_metrics["sample_path"].tolist()
                    if len(sample_paths) >= 2:
                        diversity_fraction = self._compute_diversity_fraction(
                            sample_paths,
                            rmsd_thresh=self._diversity_rmsd_thresh,
                            max_samples=self._diversity_max_samples,
                        )
                        if diversity_fraction is not None:
                            wandb_logs["diversity_fraction"] = diversity_fraction

                # Always log latest cached fractions (from full or lightweight eval)
                if self._last_designability_fraction is not None:
                    wandb_logs.setdefault("designability_fraction", self._last_designability_fraction)
                if self._last_diversity_fraction is not None:
                    wandb_logs.setdefault("diversity_fraction", self._last_diversity_fraction)

                wandb.log(wandb_logs, step=self.trained_steps)

            if torch.isnan(loss):
                if self._use_wandb:
                    wandb.alert(
                        title="Encountered NaN loss",
                        text=f"Loss NaN after {self.trained_epochs} epochs, {self.trained_steps} steps",
                    )
                raise Exception("NaN encountered")

        if return_logs:
            return global_logs

    # ----------------------------------------------------------------------------------
    # Diversity / Designability helper computations
    # ----------------------------------------------------------------------------------
    def _compute_diversity_fraction(self, pdb_paths, rmsd_thresh: float, max_samples: int = 40):
        """Compute fraction of pairwise CA RMSDs >= threshold.

        Args:
            pdb_paths: list of PDB file paths.
            rmsd_thresh: threshold in Angstrom.
            max_samples: subsample first N structures for efficiency.
        Returns:
            diversity_fraction (float) or None if computation failed.
        """
        try:
            if len(pdb_paths) < 2:
                return None
            sub_paths = pdb_paths[: max_samples]
            # Load with mdtraj
            trajs = [md.load(p) for p in sub_paths]
            # Align atom selections: take CA atoms only
            ca_trajs = [t.atom_slice([a.index for a in t.topology.atoms if a.name == "CA"]) for t in trajs]
            # Ensure consistent residue counts
            min_len = min(t.n_atoms for t in ca_trajs)
            if min_len == 0:
                return None
            trimmed = [t.atom_slice(range(min_len)) for t in ca_trajs]
            # Stack coordinates
            coords = [t.xyz[0] for t in trimmed]  # list of (N_ca,3)
            n = len(coords)
            pair_total = 0
            pair_pass = 0
            for i in range(n):
                for j in range(i + 1, n):
                    # Superpose i onto j (Kabsch) using mdtraj superpose semantics
                    # Create shallow copies to avoid modifying originals
                    ti = trimmed[i].superpose(trimmed[j])
                    rmsd_val = md.rmsd(ti, trimmed[j])[0]  # single frame
                    pair_total += 1
                    if rmsd_val >= rmsd_thresh:
                        pair_pass += 1
            if pair_total == 0:
                return None
            return pair_pass / pair_total
        except Exception as e:
            self._log.warning(f"Diversity fraction computation failed: {e}")
            return None

    def eval_fn(
        self,
        eval_dir,
        valid_loader,
        device,
        min_t=None,
        num_t=None,
        noise_scale=1.0,
        context=None,
    ):
        ckpt_eval_metrics = []
        for valid_feats, pdb_names in valid_loader:
            res_mask = du.move_to_np(valid_feats["res_mask"].bool())
            fixed_mask = du.move_to_np(valid_feats["fixed_mask"].bool())
            aatype = du.move_to_np(valid_feats["aatype"])
            gt_prot = du.move_to_np(valid_feats["atom37_pos"])
            batch_size = res_mask.shape[0]
            valid_feats = tree.map_structure(lambda x: x.to(device), valid_feats)

            # For ChiFlow, generate missing features that were removed from data loader
            if self._model_conf.model_name == "chiflow":
                # Generate dihedrals from torsion angles
                torsion_sin_cos = valid_feats['torsion_angles_sin_cos'][:, :, :3, :]  # (B, L, 3, 2) - take first 3 angles
                dihedrals = torch.atan2(torsion_sin_cos[..., 0], torsion_sin_cos[..., 1])  # (B, L, 3)
                valid_feats["dihedrals"] = dihedrals

                # Generate noisy dihedrals for inference (this replaces the missing rigids_t)
                # Use the flow matcher to create noisy version at t=1.0
                noisy_result = self._flow_matcher.dihedral_forward_marginal(
                    dihedrals, 1.0, flow_mask=None
                )
                valid_feats["dihedrals_t"] = noisy_result["dihedrals_t"]

            # Run inference
            infer_out = self.inference_fn(
                valid_feats,
                min_t=min_t,
                num_t=num_t,
                noise_scale=noise_scale,
                context=context,
            )
            final_prot = infer_out["prot_traj"][0]
            for i in range(batch_size):
                num_res = int(np.sum(res_mask[i]).item())
                unpad_fixed_mask = fixed_mask[i][res_mask[i]]
                unpad_flow_mask = 1 - unpad_fixed_mask
                unpad_prot = final_prot[i][res_mask[i]]
                unpad_gt_prot = gt_prot[i][res_mask[i]]
                unpad_gt_aatype = aatype[i][res_mask[i]]
                percent_flowed = np.sum(unpad_flow_mask) / num_res
                if self._use_ddp:
                    prot_path = os.path.join(
                        eval_dir,
                        f"len_{num_res}_sample_{i}_{self.fabric.device}_flowed_{percent_flowed:.2f}.pdb",
                    )
                else:
                    prot_path = os.path.join(
                        eval_dir,
                        f"len_{num_res}_sample_{i}_flowed_{percent_flowed:.2f}.pdb",
                    )

                # Extract argmax predicted aatype
                saved_path = au.write_prot_to_pdb(
                    unpad_prot,
                    prot_path,
                    no_indexing=True,
                    b_factors=np.tile(1 - unpad_fixed_mask[..., None], 37) * 100,
                )
                try:
                    sample_metrics = metrics.protein_metrics(
                        pdb_path=saved_path,
                        atom37_pos=unpad_prot,
                        gt_atom37_pos=unpad_gt_prot,
                        gt_aatype=unpad_gt_aatype,
                        flow_mask=unpad_flow_mask,
                    )
                except ValueError as e:
                    self._log.warning(
                        f"Failed evaluation of length {num_res} sample {i}: {e}"
                    )
                    continue
                sample_metrics["step"] = self.trained_steps
                sample_metrics["num_res"] = num_res
                sample_metrics["fixed_residues"] = np.sum(unpad_fixed_mask)
                sample_metrics["flowed_percentage"] = percent_flowed
                sample_metrics["sample_path"] = saved_path
                sample_metrics["gt_pdb"] = pdb_names[i]
                ckpt_eval_metrics.append(sample_metrics)

        # Save metrics as CSV.
        eval_metrics_csv_path = os.path.join(eval_dir, "metrics.csv")
        ckpt_eval_metrics = pd.DataFrame(ckpt_eval_metrics)
        ckpt_eval_metrics.to_csv(eval_metrics_csv_path, index=False)
        return ckpt_eval_metrics

    # ----------------------------------------------------------------------------------
    # Lightweight real-time fraction evaluation (subset of validation set)
    # ----------------------------------------------------------------------------------
    def _lightweight_fraction_eval(self, valid_loader, device):
        """Run a fast evaluation on a small subset to update fraction metrics.

        Strategy:
          - Take up to self._fraction_eval_num_samples batches (accumulating samples) until reaching target count.
          - Run inference (single reverse trajectory) to produce final structures.
          - Compute TM-score vs ground-truth for each sample (designability proxy).
          - Save temporary PDB files in a temp subdir under eval_dir (if available) or /tmp.
          - Reuse existing diversity fraction function on produced sample paths.
        """
        if self._fraction_eval_freq <= 0:
            return  # feature disabled
        try:
            import tempfile, shutil
            from tools.analysis import metrics as metrics_mod
            import mdtraj as md  # ensure available
            # Collect samples
            collected = 0
            sample_paths = []
            tm_scores = []
            temp_root = tempfile.mkdtemp(prefix="fraction_eval_")
            for valid_feats, pdb_names in valid_loader:
                if collected >= self._fraction_eval_num_samples:
                    break
                batch_size = valid_feats["res_mask"].shape[0]
                take = min(batch_size, self._fraction_eval_num_samples - collected)
                # Slice to take only needed
                slice_feats = tree.map_structure(lambda x: x[:take].to(device) if torch.is_tensor(x) else x[:take], valid_feats)
                # ChiFlow special preparation
                if self._model_conf.model_name == "chiflow":
                    torsion_sin_cos = slice_feats['torsion_angles_sin_cos'][:, :, :3, :]
                    dihedrals = torch.atan2(torsion_sin_cos[..., 0], torsion_sin_cos[..., 1])
                    slice_feats['dihedrals'] = dihedrals
                    noisy_result = self._flow_matcher.dihedral_forward_marginal(dihedrals, 1.0, flow_mask=None)
                    slice_feats['dihedrals_t'] = noisy_result['dihedrals_t']
                infer_out = self.inference_fn(slice_feats, noise_scale=self._exp_conf.noise_scale)
                final_prot = infer_out["prot_traj"][0] # (B, L, 37, 3) at t=0
                res_mask = du.move_to_np(slice_feats["res_mask"].bool())
                aatype = du.move_to_np(slice_feats["aatype"]) if "aatype" in slice_feats else None
                gt_prot = du.move_to_np(slice_feats["atom37_pos"]) if "atom37_pos" in slice_feats else None
                for i in range(take):
                    num_res = int(np.sum(res_mask[i]).item())
                    unpad_prot = final_prot[i][res_mask[i]]
                    gt_atom37 = gt_prot[i][res_mask[i]] if gt_prot is not None else None
                    prot_path = os.path.join(temp_root, f"len_{num_res}_sample_{collected+i}.pdb")
                    # Reuse writer (with minimal args)
                    au.write_prot_to_pdb(unpad_prot, prot_path, no_indexing=True)
                    sample_paths.append(prot_path)
                    if gt_atom37 is not None and aatype is not None:
                        try:
                            sample_metrics = metrics_mod.protein_metrics(
                                pdb_path=prot_path,
                                atom37_pos=unpad_prot,
                                gt_atom37_pos=gt_atom37,
                                gt_aatype=aatype[i][res_mask[i]],
                                flow_mask=np.ones(num_res),
                            )
                            if 'tm_score' in sample_metrics:
                                tm_scores.append(sample_metrics['tm_score'])
                        except Exception:
                            pass
                collected += take
            # Designability fraction
            if tm_scores:
                self._last_designability_fraction = float(
                    (np.array(tm_scores) >= self._designability_tm_thresh).mean()
                )
            # Diversity fraction
            if len(sample_paths) >= 2:
                diversity_fraction = self._compute_diversity_fraction(
                    sample_paths,
                    rmsd_thresh=self._diversity_rmsd_thresh,
                    max_samples=min(self._diversity_max_samples, len(sample_paths)),
                )
                if diversity_fraction is not None:
                    self._last_diversity_fraction = float(diversity_fraction)
            # Cleanup
            shutil.rmtree(temp_root, ignore_errors=True)
        except Exception as e:
            self._log.warning(f"Lightweight fraction eval failed: {e}")
            return

    def _prepare_chiflow_batch(self, batch):
        """Prepare batch data for ChiFlow model.

        Convert torsion_angles_sin_cos to dihedrals format expected by ChiFlow.
        """
        if "torsion_angles_sin_cos" in batch:
            # Extract phi, psi, omega from torsion_angles_sin_cos
            # Shape: (batch, seq, 7, 2) -> take first 3 angles (phi, psi, omega)
            phi_psi_omega_sin_cos = batch["torsion_angles_sin_cos"][:, :, :3, :]  # (B, L, 3, 2)

            # Convert sin/cos to angles
            dihedrals = torch.atan2(
                phi_psi_omega_sin_cos[:, :, :, 0],  # sin
                phi_psi_omega_sin_cos[:, :, :, 1]   # cos
            )  # (B, L, 3)

            # Add dihedrals to batch
            batch["dihedrals"] = dihedrals

        return batch

    def _chiflow_loss_fn(self, batch):
        """ChiFlow analytic torus flow matching loss.

        Implements linear torus path x_t = wrap(x0 + t*(x1-x0)) with x1 random noise (target angles).
        Supervises vector field v_theta(x_t,t) against ground truth constant field (x1-x0) along path.
        Per-sample t ~ U(0,1). Optionally includes chirality mirror penalty and backbone reconstruction.
        """
        from foldflow.models.chiflow import angle_diff, linear_torus_path

        gt_dihedrals = batch['dihedrals']  # (B,L,3)
        res_mask = batch['res_mask']
        flow_mask = 1 - batch.get('fixed_mask', torch.zeros_like(res_mask))
        loss_mask = (res_mask * flow_mask).float()
        device = gt_dihedrals.device
        B, L, D = gt_dihedrals.shape

        # Sample per-sample time t ~ U(0,1)
        t = torch.rand(B, device=device).view(B, 1, 1)

        # Sample random target endpoint x1 uniformly on torus
        x1 = torch.rand_like(gt_dihedrals) * 2 * torch.pi - torch.pi

        # Construct interpolation
        x0 = gt_dihedrals.detach()  # treat as data
        x_t = linear_torus_path(x0, x1, t)

        # Ground truth vector field along linear path is constant difference scaled by wrap
        gt_v = angle_diff(x1, x0)  # (B,L,3)
        # (independent of t for linear interpolation) — optionally scale by 1 for direct supervision

        # Predict vector field at x_t
        model_out = self.model.flow_matcher.forward({**batch, 'dihedrals': x0}, x_t=x_t, compute_mirror_loss=False)
        pred_v = model_out['dihedral_flow']  # (B,L,3)

        # MSE with masking
        mse = (pred_v - gt_v) ** 2 * loss_mask.unsqueeze(-1)
        dihedral_flow_loss = torch.sum(mse) / (loss_mask.sum() * D + 1e-10)

        # Mirror chirality warm-up (increase weight after warmup steps)
        mirror_constraint_loss = torch.tensor(0.0, device=device)
        mirror_warmup = getattr(self._exp_conf, 'mirror_warmup_steps', 0)
        base_mirror_weight = getattr(self._model_conf, 'mirror_constraint_weight', 0.0)
        if base_mirror_weight > 0.0:
            # Compute current mirror loss using original batch (x0) if model supports it
            mirror_out = self.model.flow_matcher.forward(batch, compute_mirror_loss=True)
            if hasattr(self.model.flow_matcher, 'mirror_loss'):
                progress = min(1.0, self.trained_steps / max(1, mirror_warmup)) if mirror_warmup > 0 else 1.0
                mirror_constraint_loss = self.model.flow_matcher.mirror_loss * base_mirror_weight * progress
        total_loss = dihedral_flow_loss + mirror_constraint_loss

        backbone_loss = torch.tensor(0.0, device=device)
        if hasattr(self._model, 'reconstruct_backbone') and getattr(self._exp_conf, 'backbone_recon_weight', 0.0) > 0:
            with torch.no_grad():
                gt_backbone = self._model.reconstruct_backbone(gt_dihedrals, batch)
            # Optionally reconstruct from x_t (closer to training distribution) or predicted angles (x0 here)
            pred_backbone = self._model.reconstruct_backbone(gt_dihedrals, batch)
            bb_mse = (gt_backbone - pred_backbone) ** 2 * loss_mask.unsqueeze(-1).unsqueeze(-1)
            backbone_loss = torch.sum(bb_mse) / (loss_mask.sum() * 3 + 1e-10)
            backbone_loss *= getattr(self._exp_conf, 'backbone_recon_weight', 0.0)
            total_loss += backbone_loss

        aux_data = {
            'batch_train_loss': total_loss.detach(),
            'dihedral_flow_loss': dihedral_flow_loss.detach(),
            'mirror_constraint_loss': mirror_constraint_loss.detach(),
            'backbone_loss': backbone_loss.detach(),
            'total_loss': total_loss.detach(),
            'examples_per_step': torch.tensor(B, device=device),
            'res_length': torch.mean(torch.sum(res_mask, dim=-1).float()),
            't_mean': t.mean().detach(),
        }
        return total_loss, aux_data

    def _self_conditioning(self, batch):
        model_sc = self.model(batch)
        batch["sc_ca_t"] = model_sc["rigids"][..., 4:]
        return batch

    def loss_fn(self, batch):
        """Computes loss and auxiliary data.

        Args:
            batch: Batched data.
            model_out: Output of model ran on batch.

        Returns:
            loss: Final training loss scalar.
            aux_data: Additional logging data.
        """
        # Handle ChiFlow data format conversion
        if self._model_conf.model_name == "chiflow":
            batch = self._prepare_chiflow_batch(batch)
            return self._chiflow_loss_fn(batch)

        if (
            self._model_conf.embed.embed_self_conditioning
            and self.trained_steps % 2 == 1
        ):
            # if self._model_conf.embed.embed_self_conditioning and random.random() > 0.5:
            with torch.no_grad():
                batch = self._self_conditioning(batch)

        _, gt_rot_u_t = self._flow_matcher._so3_fm.vectorfield(
            batch["rot_vectorfield"], batch["rot_t"], batch["t"]
        )

        model_out = self.model(batch)
        bb_mask = batch["res_mask"]
        flow_mask = 1 - batch["fixed_mask"]
        loss_mask = bb_mask * flow_mask
        batch_size, num_res = bb_mask.shape

        gt_trans_u_t = batch["trans_vectorfield"]
        rot_vectorfield_scaling = batch["rot_vectorfield_scaling"]
        trans_vectorfield_scaling = batch["trans_vectorfield_scaling"]
        batch_loss_mask = torch.any(bb_mask, dim=-1)

        pred_rot_v_t = model_out["rot_vectorfield"] * flow_mask[..., None, None]
        pred_trans_v_t = model_out["trans_vectorfield"] * flow_mask[..., None]

        # Translation vectorfield loss
        trans_vectorfield_mse = (gt_trans_u_t - pred_trans_v_t) ** 2 * loss_mask[
            ..., None
        ]
        trans_vectorfield_loss = torch.sum(
            trans_vectorfield_mse / trans_vectorfield_scaling[:, None, None] ** 2,
            dim=(-1, -2),
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        # Translation x0 loss
        gt_trans_x0 = batch["rigids_0"][..., 4:] * self._exp_conf.coordinate_scaling
        pred_trans_x0 = model_out["rigids"][..., 4:] * self._exp_conf.coordinate_scaling
        trans_x0_loss = torch.sum(
            (gt_trans_x0 - pred_trans_x0) ** 2 * loss_mask[..., None], dim=(-1, -2)
        ) / (loss_mask.sum(dim=-1) + 1e-10)

        trans_loss = trans_vectorfield_loss * (
            batch["t"] > self._exp_conf.trans_x0_threshold
        ) + trans_x0_loss * (batch["t"] <= self._exp_conf.trans_x0_threshold)
        trans_loss *= self._exp_conf.trans_loss_weight
        trans_loss *= int(self._fm_conf.flow_trans)

        # Rotation loss
        # gt_rot_u_t and pred_rot_v_t are matrices convert
        t_shape = batch["rot_t"].shape[0]
        rot_t = rearrange(batch["rot_t"], "t n c d -> (t n) c d", c=3, d=3).double()
        gt_rot_u_t = rearrange(gt_rot_u_t, "t n c d -> (t n) c d", c=3, d=3)
        pred_rot_v_t = rearrange(pred_rot_v_t, "t n c d -> (t n) c d", c=3, d=3)
        try:
            rot_t = rot_t.double()
            gt_at_id = pt_to_identity(rot_t, gt_rot_u_t)
            gt_rot_u_t = hat_inv(gt_at_id)
            pred_at_id = pt_to_identity(rot_t, pred_rot_v_t)
            pred_rot_v_t = hat_inv(pred_at_id)
        except ValueError as e:
            self._log.info(
                f"Skew symmetric error gt {((gt_at_id + gt_at_id.transpose(-1, -2))**2).mean()} "
                f"pred {((pred_at_id + pred_at_id.transpose(-1, -2))**2).mean()} Skipping rot loss"
            )
            gt_rot_u_t = torch.zeros_like(rot_t[..., 0])
            pred_rot_v_t = torch.zeros_like(rot_t[..., 0])

        gt_rot_u_t = rearrange(gt_rot_u_t, "(t n) c -> t n c", t=t_shape, c=3)
        pred_rot_v_t = rearrange(pred_rot_v_t, "(t n) c -> t n c", t=t_shape, c=3)

        if self._exp_conf.separate_rot_loss:
            gt_rot_angle = torch.norm(gt_rot_u_t, dim=-1, keepdim=True)
            gt_rot_axis = gt_rot_u_t / (gt_rot_angle + 1e-6)

            pred_rot_angle = torch.norm(pred_rot_v_t, dim=-1, keepdim=True)
            pred_rot_axis = pred_rot_v_t / (pred_rot_angle + 1e-6)

            # Separate loss on the axis
            axis_loss = (gt_rot_axis - pred_rot_axis) ** 2 * loss_mask[..., None]
            axis_loss = torch.sum(axis_loss, dim=(-1, -2)) / (
                loss_mask.sum(dim=-1) + 1e-10
            )

            # Separate loss on the angle
            angle_loss = (gt_rot_angle - pred_rot_angle) ** 2 * loss_mask[..., None]
            angle_loss = torch.sum(
                angle_loss / rot_vectorfield_scaling[:, None, None] ** 2, dim=(-1, -2)
            ) / (loss_mask.sum(dim=-1) + 1e-10)
            angle_loss *= self._exp_conf.rot_loss_weight
            angle_loss *= batch["t"] > self._exp_conf.rot_loss_t_threshold
            rot_loss = angle_loss + axis_loss
        else:
            rot_mse = (gt_rot_u_t - pred_rot_v_t) ** 2 * loss_mask[..., None]
            rot_loss = torch.sum(
                rot_mse / rot_vectorfield_scaling[:, None, None] ** 2,
                dim=(-1, -2),
            ) / (loss_mask.sum(dim=-1) + 1e-10)
            rot_loss *= self._exp_conf.rot_loss_weight
            rot_loss *= batch["t"] > self._exp_conf.rot_loss_t_threshold
        rot_loss *= int(self._fm_conf.flow_rot)

        # Backbone atom loss
        pred_atom37 = model_out["atom37"][:, :, :5]
        gt_rigids = ru.Rigid.from_tensor_7(batch["rigids_0"].type(torch.float32))
        gt_psi = batch["torsion_angles_sin_cos"][..., 2, :]
        gt_atom37, atom37_mask, _, _ = all_atom.compute_backbone(gt_rigids, gt_psi)
        gt_atom37 = gt_atom37[:, :, :5]
        atom37_mask = atom37_mask[:, :, :5]

        gt_atom37 = gt_atom37.to(pred_atom37.device)
        atom37_mask = atom37_mask.to(pred_atom37.device)
        bb_atom_loss_mask = atom37_mask * loss_mask[..., None]
        bb_atom_loss = torch.sum(
            (pred_atom37 - gt_atom37) ** 2 * bb_atom_loss_mask[..., None],
            dim=(-1, -2, -3),
        ) / (bb_atom_loss_mask.sum(dim=(-1, -2)) + 1e-10)
        bb_atom_loss *= self._exp_conf.bb_atom_loss_weight
        bb_atom_loss *= batch["t"] < self._exp_conf.bb_atom_loss_t_filter
        bb_atom_loss *= self._exp_conf.aux_loss_weight

        # Pairwise distance loss
        gt_flat_atoms = gt_atom37.reshape([batch_size, num_res * 5, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1
        )
        pred_flat_atoms = pred_atom37.reshape([batch_size, num_res * 5, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1
        )

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 5))
        flat_loss_mask = flat_loss_mask.reshape([batch_size, num_res * 5])
        flat_res_mask = torch.tile(bb_mask[:, :, None], (1, 1, 5))
        flat_res_mask = flat_res_mask.reshape([batch_size, num_res * 5])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        # No loss on anything >6A
        proximity_mask = gt_pair_dists < 6
        pair_dist_mask = pair_dist_mask * proximity_mask

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists) ** 2 * pair_dist_mask, dim=(1, 2)
        )
        dist_mat_loss /= torch.sum(pair_dist_mask, dim=(1, 2)) - num_res
        dist_mat_loss *= self._exp_conf.dist_mat_loss_weight
        dist_mat_loss *= batch["t"] < self._exp_conf.dist_mat_loss_t_filter
        dist_mat_loss *= self._exp_conf.aux_loss_weight

        final_loss = rot_loss + trans_loss + bb_atom_loss + dist_mat_loss

        def normalize_loss(x):
            return x.sum() / (batch_loss_mask.sum() + 1e-10)

        aux_data = {
            "batch_train_loss": final_loss,
            "batch_rot_loss": rot_loss,
            "batch_trans_loss": trans_loss,
            "batch_bb_atom_loss": bb_atom_loss,
            "batch_dist_mat_loss": dist_mat_loss,
            "total_loss": normalize_loss(final_loss),
            "rot_loss": normalize_loss(rot_loss),
            "trans_loss": normalize_loss(trans_loss),
            "bb_atom_loss": normalize_loss(bb_atom_loss),
            "dist_mat_loss": normalize_loss(dist_mat_loss),
            "examples_per_step": torch.tensor(batch_size),
            "res_length": torch.mean(torch.sum(bb_mask, dim=-1)),
        }

        # Maintain a history of the past N number of steps.
        # Helpful for debugging.
        self._aux_data_history.append(
            {"aux_data": aux_data, "model_out": model_out, "batch": batch}
        )

        assert final_loss.shape == (batch_size,)
        assert batch_loss_mask.shape == (batch_size,)
        return normalize_loss(final_loss), aux_data

    def _calc_trans_0(self, trans_vectorfield, trans_t, t):
        beta_t = self._flow_matcher._se3_fm._r3_fm.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        cond_var = 1 - torch.exp(-beta_t)
        return (trans_vectorfield * cond_var + trans_t) / torch.exp(-1 / 2 * beta_t)

    def _set_t_feats(self, feats, t, t_placeholder):
        feats["t"] = t * t_placeholder
        (
            rot_vectorfield_scaling,
            trans_vectorfield_scaling,
        ) = self.flow_matcher.vectorfield_scaling(t)
        feats["rot_vectorfield_scaling"] = rot_vectorfield_scaling * t_placeholder
        feats["trans_vectorfield_scaling"] = trans_vectorfield_scaling * t_placeholder
        return feats

    def forward_traj(self, x_0, min_t, num_t):
        forward_steps = np.linspace(min_t, 1.0, num_t)[:-1]
        x_traj = [x_0]
        for t in forward_steps:
            x_t = self.flow_matcher.se3_fm._r3_fm.forward(x_traj[-1], t, num_t)
            x_traj.append(x_t)
        x_traj = torch.stack(x_traj, axis=0)
        return x_traj

    def inference_fn(
        self,
        data_init,
        num_t=None,
        min_t=None,
        center=True,
        aux_traj=False,
        self_condition=True,
        noise_scale=1.0,
        context=None,
    ):
        """Inference function.

        Args:
            data_init: Initial data values for sampling.
        """

        # Handle ChiFlow separately
        if self._model_conf.model_name == "chiflow":
            return self._chiflow_inference_fn(
                data_init, num_t, min_t, noise_scale
            )

        # Run reverse process.
        sample_feats = copy.deepcopy(data_init)
        device = sample_feats["rigids_t"].device
        if sample_feats["rigids_t"].ndim == 2:
            t_placeholder = torch.ones((1,)).to(device)
        else:
            t_placeholder = torch.ones((sample_feats["rigids_t"].shape[0],)).to(device)
        if num_t is None:
            num_t = self._data_conf.num_t
        if min_t is None:
            min_t = self._data_conf.min_t
        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = reverse_steps[0] - reverse_steps[1]
        # dt = 1/num_t
        all_rigids = [du.move_to_np(copy.deepcopy(sample_feats["rigids_t"]))]
        all_bb_prots = []
        all_trans_0_pred = []
        all_bb_0_pred = []
        with torch.no_grad():
            if self._model_conf.embed.embed_self_conditioning and self_condition:
                sample_feats = self._set_t_feats(
                    sample_feats, reverse_steps[0], t_placeholder
                )
                sample_feats = self._self_conditioning(sample_feats)
            for t in reverse_steps:

                sample_feats = self._set_t_feats(sample_feats, t, t_placeholder)
                model_out = self.model(sample_feats)
                rot_vectorfield = model_out["rot_vectorfield"]
                trans_vectorfield = model_out["trans_vectorfield"]
                rigid_pred = model_out["rigids"]
                if self._model_conf.embed.embed_self_conditioning:
                    sample_feats["sc_ca_t"] = rigid_pred[..., 4:]
                fixed_mask = sample_feats["fixed_mask"] * sample_feats["res_mask"]
                flow_mask = (1 - sample_feats["fixed_mask"]) * sample_feats["res_mask"]
                rots_t, trans_t, rigids_t = self.flow_matcher.reverse(
                    rigid_t=ru.Rigid.from_tensor_7(sample_feats["rigids_t"]),
                    rot_vectorfield=du.move_to_np(rot_vectorfield),
                    trans_vectorfield=du.move_to_np(trans_vectorfield),
                    flow_mask=du.move_to_np(flow_mask),
                    t=t,
                    dt=dt,
                    center=center,
                    noise_scale=noise_scale,
                )


                sample_feats["rigids_t"] = rigids_t.to_tensor_7().to(device)
                if aux_traj:
                    all_rigids.append(du.move_to_np(rigids_t.to_tensor_7()))

                # Calculate x0 prediction derived from vectorfield predictions.
                gt_trans_0 = sample_feats["rigids_t"][..., 4:]
                pred_trans_0 = rigid_pred[..., 4:]
                trans_pred_0 = (
                    flow_mask[..., None] * pred_trans_0
                    + fixed_mask[..., None] * gt_trans_0
                )
                psi_pred = model_out["psi"]
                if aux_traj:
                    atom37_0 = all_atom.compute_backbone(
                        ru.Rigid.from_tensor_7(rigid_pred), psi_pred
                    )[0]
                    all_bb_0_pred.append(du.move_to_np(atom37_0))
                    all_trans_0_pred.append(du.move_to_np(trans_pred_0))
                atom37_t = all_atom.compute_backbone(rigids_t, psi_pred)[0]
                all_bb_prots.append(du.move_to_np(atom37_t))

        # Flip trajectory so that it starts from t=0.
        # This helps visualization.
        flip = lambda x: np.flip(np.stack(x), (0,))
        all_bb_prots = flip(all_bb_prots)
        if aux_traj:
            all_rigids = flip(all_rigids)
            all_trans_0_pred = flip(all_trans_0_pred)
            all_bb_0_pred = flip(all_bb_0_pred)

        ret = {
            "prot_traj": all_bb_prots,
        }
        if aux_traj:
            ret["rigid_traj"] = all_rigids
            ret["trans_traj"] = all_trans_0_pred
            ret["psi_pred"] = psi_pred[None]
            ret["rigid_0_traj"] = all_bb_0_pred
        return ret

    def _chiflow_inference_fn(self, data_init, num_t=None, min_t=None, noise_scale=1.0):
        """ChiFlow-specific inference function for dihedral-based flow matching."""
        from foldflow.models.flows.common.nerf import nerf_build_batch

        # Create a shallow copy and detach tensors to avoid deepcopy issues
        sample_feats = {}
        for k, v in data_init.items():
            if isinstance(v, torch.Tensor):
                sample_feats[k] = v.detach().clone()
            else:
                sample_feats[k] = v

        device = sample_feats["dihedrals_t"].device

        if num_t is None:
            num_t = self._data_conf.num_t
        if min_t is None:
            min_t = self._data_conf.min_t

        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = reverse_steps[0] - reverse_steps[1] if num_t > 1 else 1.0

        all_dihedrals = [du.move_to_np(copy.deepcopy(sample_feats["dihedrals_t"]))]
        all_bb_prots = []

        with torch.no_grad():
            for t in reverse_steps:
                # Set time features
                t_tensor = torch.tensor([t], device=device).expand(sample_feats["dihedrals_t"].shape[0])
                sample_feats["t"] = t_tensor

                # Model forward pass
                model_out = self.model(sample_feats)
                dihedral_vectorfield = model_out["dihedral_flow"]

                # Euler integration step (reverse direction)
                sample_feats["dihedrals_t"] = sample_feats["dihedrals_t"] + dihedral_vectorfield * dt

                # Project to torus [-π, π]
                sample_feats["dihedrals_t"] = torch.remainder(
                    sample_feats["dihedrals_t"] + torch.pi, 2 * torch.pi
                ) - torch.pi

                all_dihedrals.append(du.move_to_np(sample_feats["dihedrals_t"]))

                # Convert to backbone coordinates
                phi = sample_feats["dihedrals_t"][:, :, 0]
                psi = sample_feats["dihedrals_t"][:, :, 1]
                omega = sample_feats["dihedrals_t"][:, :, 2]

                # Build backbone coordinates (returns shape: batch, length, 3, 3)
                backbone_coords = nerf_build_batch(phi, psi, omega)
                # No need to slice - use all coordinates

                # Convert to atom37 format (batch, length, 37, 3)
                atom37_coords = torch.zeros(phi.shape[0], phi.shape[1], 37, 3, device=device)
                # Map backbone atoms: N=0, CA=1, C=2 in atom37
                atom37_coords[:, :, 0, :] = backbone_coords[:, :, 0, :]  # N
                atom37_coords[:, :, 1, :] = backbone_coords[:, :, 1, :]  # CA
                atom37_coords[:, :, 2, :] = backbone_coords[:, :, 2, :]  # C

                all_bb_prots.append(du.move_to_np(atom37_coords))

        # Flip trajectory so that it starts from t=0
        flip = lambda x: np.flip(np.stack(x), (0,))
        all_bb_prots = flip(all_bb_prots)
        all_dihedrals = flip(all_dihedrals)

        return {
            "prot_traj": all_bb_prots,
            "dihedral_traj": all_dihedrals,
        }


@hydra.main(version_base=None, config_path="config/", config_name="base")
def run(conf: DictConfig) -> None:

    # Fixes bug in https://github.com/wandb/wandb/issues/1525
    os.environ["WANDB_START_METHOD"] = "thread"
    exp = Experiment(conf=conf)
    return exp.start_training()


if __name__ == "__main__":
    run()
