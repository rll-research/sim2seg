# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from dm_env import StepType, specs

import dmc
from dmc import ExtendedTimeStep
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrajVideoRecorder
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

NUM_CLASSES = 6

def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = [(3, 256, 256), (2,), (10, 3)]
    cfg.action_shape = (2,)
    return hydra.utils.instantiate(cfg)

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        self._global_step = 0
        self._global_episode = 0
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(
            self.eval_env.observation_spec(),
            self.eval_env.action_spec(),
            self.cfg.agent
        )
        self.train_env.agent = self.agent
        self.eval_env.agent = self.agent

        self.timer = utils.Timer()

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)

        if self.cfg.use_s2s:
            from models import load_model
            self.s2s_model = load_model(self.cfg.s2s_mode, self.cfg.s2s_device)
        else:
            self.s2s_model = None

        # create envs
        self.BASE_FOLDER = "BASE_PATH_TO_EXECUTABLE"
        self.setup_train_env()
        self.setup_eval_env()

        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,) + (2,), np.float32, 'desired_goal'),
                      specs.Array((1,) + (2,), np.float32, 'achieved_goal'),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'),
                      specs.Array((1, self.cfg.num_odom,) + (3,),
                            np.float32,"odom",),
                      specs.Array((1, 3), np.float32, name="rot"),
                      specs.Array((1, 1), np.int32, name="goalId")
                      )
        self.replay_storage = ReplayBufferStorage(data_specs, self.work_dir / 'buffer', can=self.cfg.use_s2s)

        self.replay_loader = make_replay_loader(
            self.work_dir / "buffer",
            self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot,
            self.cfg.nstep,
            self.cfg.discount,
            self.cfg.lambda_steer,
            self.cfg.lambda_accel,
            self.cfg.lambda_upright,
            self.cfg.lambda_prox,
            self.cfg.her_ratio,
            self.cfg.use_s2s,
            self.cfg.lambda_lp
        )
        self._replay_iter = None


    def setup_train_env(self):
        print(f"setting up train env {self.cfg.train_task_name}")
        self.train_env = dmc.unity_lstm_make(f"{self.BASE_FOLDER}/{self.cfg.train_task_name}",
                                       self.cfg.frame_stack,
                                       self.cfg.action_repeat,
                                       self.cfg.seed + self._global_episode,
                                       self.cfg.base_port + 100,
                                       num_odom=self.cfg.num_odom,
                                       can=self.cfg.use_s2s,
                                       s2s=self.s2s_model,
                                       num_actions=self.cfg.agent.num_actions,
                                       seq_goal=self.cfg.seq_goal)
        self.video_recorder = TrajVideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )

    def setup_eval_env(self):
        print("setting up eval env", self.cfg.eval_task_name)
        self.eval_env = dmc.unity_lstm_make(
            f"{self.BASE_FOLDER}/{self.cfg.eval_task_name}",
            self.cfg.frame_stack,
            self.cfg.action_repeat,
            self.cfg.seed,
            self.cfg.base_port,
            num_odom=self.cfg.num_odom,
            can=self.cfg.use_s2s,
            s2s=self.s2s_model,
            num_actions=self.cfg.agent.num_actions,
            final_eval=self.cfg.final_eval
        )
        self.video_recorder = TrajVideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )

    def switch_env_and_s2s(self):
        from models import load_model

        depth_str = ""
        if "depth" in self.cfg.s2s_mode:
            depth_str = "_depth"
        if "meadow" in self.cfg.s2s_mode:
            self.cfg.s2s_mode = "canyon" + depth_str
            self.cfg.train_task_name = self.cfg.train_names[1]
            self.cfg.eval_task_name = self.cfg.eval_names[1]
        elif "canyon" in self.cfg.s2s_mode:
            self.cfg.s2s_mode = "rl" + depth_str
            self.cfg.train_task_name = self.cfg.train_names[2]
            self.cfg.eval_task_name = self.cfg.eval_names[2]
        elif "rl" in self.cfg.s2s_mode:
            self.cfg.s2s_mode = "meadow" + depth_str
            self.cfg.train_task_name = self.cfg.train_names[0]
            self.cfg.eval_task_name = self.cfg.eval_names[0]
        else:
            raise NotImplementedError

        self.s2s_model = load_model(self.cfg.s2s_mode, self.cfg.meadow_device)
        self.train_env.close()
        self.setup_train_env()
        if not self.cfg.final_eval:
            # don't switch between test envs is final_eval
            self.eval_env.close()
            self.setup_eval_env()

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    @property
    def real_replay_iter(self):
        if self._real_replay_iter is None:
            self._real_replay_iter = iter(self.real_replay_loader)
        return self._real_replay_iter

    def eval(self):
        step, episode, total_reward, n_goals, n_complete = 0, 0, 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    if self.cfg.use_s2s:
                        obs = time_step.observation
                        if time_step.observation.shape[0] == 1:
                            obs = time_step.observation[0]
                        obs = np.argmax(obs, axis=0)
                        obs_onehot = (
                            np.eye(NUM_CLASSES)[obs] 
                            .transpose((2, 0, 1))
                            .astype(time_step.observation.dtype)
                        )
                        if time_step.observation.shape[0] == 1:
                            obs_onehot = np.array([obs_onehot])
                    else:
                        obs_onehot = time_step.observation
                    action = self.agent.act(
                        torch.as_tensor(obs_onehot, device=self.device),
                        torch.as_tensor(time_step.desired_goal, device=self.device),
                        torch.as_tensor(time_step.odom, device=self.device),
                        self.global_step,
                        eval_mode=True,
                    )
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f"{self.global_frame}.mp4")
            n_goals += (time_step.info["goalId"] + 1)
            n_complete += time_step.info["completedGoals"]

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            log("episode_reward", total_reward / episode)
            log("episode_length", step * self.cfg.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)
            log("n_goals", n_goals / episode)
            log("n_complete", n_complete / episode)
            log("percent_complete", n_complete / n_goals)


    def train(self):
        # predicates
        train_until_step = utils.Until(
            self.cfg.num_train_frames, self.cfg.action_repeat
        )
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(
            self.cfg.eval_every_frames, self.cfg.action_repeat
        )
        switch_every_step = utils.Every(
            self.cfg.switch_every_frames, self.cfg.action_repeat
        )
        save_every_step = utils.Every(
            self.cfg.save_every_frames, self.cfg.action_repeat
        )

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        metrics = None
        while train_until_step(self.global_step):
            # if training on all 3 envs, switch when necessary
            if self.cfg.use_switch_every and switch_every_step(self.global_step):
                self.switch_env_and_s2s()
                time_step = self.train_env.reset()

            if time_step.last():
                self._global_episode += 1
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(
                        self.global_frame, ty="train"
                    ) as log:
                        log("fps", episode_frame / elapsed_time)
                        log("total_time", total_time)
                        log("episode_reward", episode_reward.sum())
                        log("episode_length", episode_frame)
                        log("episode", self.global_episode)
                        log("buffer_size", len(self.replay_storage))
                        log("step", self.global_step)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                episode_step = 0
                episode_reward = 0

            # evaluate
            if eval_every_step(self.global_step):
                self.logger.log(
                    "eval_total_time", self.timer.total_time(), self.global_frame
                )
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(
                    torch.as_tensor(time_step.observation, device=self.device),
                    torch.as_tensor(time_step.achieved_goal, device=self.device),
                    torch.as_tensor(time_step.odom, device=self.device),
                    self.global_step,
                    eval_mode=False,
                )

            # try to update the agent
            if not seed_until_step(self.global_step):
                for i in range(self.cfg.n_update):
                    metrics = self.agent.update(self.replay_iter, self.global_step)
                    if i == self.cfg.n_update - 1:
                        self.logger.log_metrics(metrics, self.global_frame, ty="train")

            if save_every_step(self.global_step):
                self.save_snapshot()

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            episode_step += 1
            self._global_step += 1
            torch.cuda.empty_cache()

    def save_snapshot(self):
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f"snapshot_{self.global_frame}.pt"
        keys_to_save = ["agent", "timer", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open("wb") as f:
            torch.save(payload, f)

    def load_snapshot(self, snapshot):
        with snapshot.open("rb") as f:
            payload = torch.load(f)
        for k, v in payload.items():
            if k == "agent":
                self.__dict__[k] = v


@hydra.main(config_path="cfgs", config_name="config")
def main(cfg):
    from train import Workspace as W

    if cfg.use_wandb:
        run = wandb.init(
            entity="ucbcal",
            name=f"{cfg.experiment_name}",
            project="sim2real_drq",
            sync_tensorboard=True,
        )
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = Path(cfg.restore_snapshot_path)
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot(snapshot)
    workspace.train()


if __name__ == "__main__":

    main()
