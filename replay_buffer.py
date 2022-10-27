# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import io
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
import torch.nn.functional as F
import einops
from einops import rearrange, reduce

NUM_CLASSES = 6

def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1

def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())

def load_episode(fn):
    with fn.open("rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode

class ReplayBufferStorage:
    def __init__(self, data_specs, replay_dir, can=True):
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        self.can = can
        if self.can:
            print("buffer using can")
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, time_step):
        for spec in self._data_specs:
            if spec.name == "goalId":
                value = time_step.info[spec.name]
            else:
                value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            if spec.name == "goalId":
                value = value.reshape((1, 1)).astype(spec.dtype)
            if spec.name == "observation" and self.can:
                # argmax on the channels dimension, (.. x H x W x C)
                # first, transpose to put in the back, argmax for the dimension, then transpose back.
                # mainly for compatibility with the buffer
                value = torch.transpose(torch.as_tensor(value), -1, -3)
                value = torch.argmax(value, dim=-1, keepdim=True)
                value = torch.transpose(value, -1, -3).numpy().astype(np.uint8)
            assert (
                spec.shape == value.shape and spec.dtype == value.dtype
            ), f"{spec.name} should be {spec.shape} and {spec.dtype}, but is {value.shape}, {value.dtype}"
            self._current_episode[spec.name].append(value)
        if time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob("*.npz"):
            _, _, eps_len = fn.stem.split("_")
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        eps_fn = f"{ts}_{eps_idx}_{eps_len}.npz"
        save_episode(episode, self._replay_dir / eps_fn)


class ReplayBuffer(IterableDataset):
    def __init__(
        self,
        replay_dir,
        max_size,
        num_workers,
        nstep,
        discount,
        fetch_every,
        save_snapshot,
        lambda_steer,
        lambda_accel,
        lambda_upright,
        lambda_prox,
        her_ratio=0,
        can=True,
        lambda_lp=1
    ):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self.her_ratio = her_ratio
        self.goal_dist = 2

        # reward shaping
        self.lambda_steer = lambda_steer
        self.lambda_accel = lambda_accel
        self.lambda_upright = lambda_upright
        self.lambda_prox = lambda_prox
        self.lambda_lp = lambda_lp
        self.can = can

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self, bypass=False):
        if self._samples_since_last_fetch < self._fetch_every and not bypass:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob("*.npz"), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep) + 1  #  + 1) + 1

        achieved_goal_arr = episode['achieved_goal']
        if np.random.random() < self.her_ratio:
            goal_idx = np.random.randint(idx, episode_len(episode) - self._nstep + 1)
            goal_world = episode['desired_goal'][goal_idx]
            achieved_goal_arr = episode['achieved_goal'].copy() # todo: no need to copy entire
            for i in range(-1, self._nstep+1):
                ego_world = episode['desired_goal'][idx + i]
                theta = episode['rot'][idx + i][..., 1]
                achieved_goal_arr[idx + i] = self.convert_coordinates(goal_world, ego_world, theta)
            s_a = achieved_goal_arr[idx - 1]
            ns_a = achieved_goal_arr[idx + self._nstep - 1]
        else:
            goal_idx = idx
            s_a = episode['achieved_goal'][idx - 1]
            ns_a = episode['achieved_goal'][idx + self._nstep - 1]

        if not self.can:
            obs = episode['observation'][idx - 1]
        else:
            t = (
                torch.as_tensor(episode["observation"][idx - 1])
                .transpose(-1, -3)
                .long()
            )
            obs = (
                F.one_hot(t.squeeze(-1), num_classes=NUM_CLASSES)
                .transpose(-1, -3)
                .float()
                .numpy()
            )
        action = episode["action"][idx]
        if not self.can:
            next_obs = episode["observation"][idx + self._nstep - 1]
        else:
            next_t = (
                torch.as_tensor(episode["observation"][idx + self._nstep - 1])
                .transpose(-1, -3)
                .long()
            )
            next_obs = (
                F.one_hot(
                    next_t.squeeze(-1), num_classes=NUM_CLASSES
                )
                .transpose(-1, -3)
                .float()
                .numpy()
            )
        reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["discount"][idx])
        odom = episode["odom"][idx - 1]
        nodom = episode["odom"][idx + self._nstep - 1]

        reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["discount"][idx])

        completed_goal = False
        for i in range(self._nstep):
            # achieved_goal_arr to account for relabelling
            dist_norm = np.linalg.norm(achieved_goal_arr[idx + i + 1], axis=-1)
            goal_reward = (dist_norm < self.goal_dist).astype(float) * 101 - 1 * self.lambda_lp + episode['reward'][idx + i]
            prox_reward = self.lambda_prox * max(0, 10-dist_norm)

            # penalize x, z rotations
            upright_penalty = (
                np.minimum(
                    episode["rot"][idx + i][..., 0:3:2],
                    360 - episode["rot"][idx + i][..., 0:3:2],
                )
                / 180
            )
            upright_reward = -self.lambda_upright * upright_penalty.mean(
                axis=-1
            )  # always between 0, -5

            steer_reward = (-self.lambda_steer) * np.absolute(
                episode["action"][idx + i][..., 0]
            )
            accel_reward = self.lambda_accel * np.square(
                episode["action"][idx + i][..., 1]
            )

            steer_reward = steer_reward.mean(axis=-1)
            accel_reward = accel_reward.mean(axis=-1)

            completed_goal = completed_goal or (dist_norm < self.goal_dist)
            if (episode['goalId'][idx+i][0] != episode['goalId'][idx][0]):
                if completed_goal:
                    step_reward = 100
                else:
                    step_reward = 0
            else:
                step_reward = goal_reward + prox_reward + upright_reward + steer_reward + accel_reward

            reward += discount * step_reward
            discount *= episode["discount"][idx + i] * self._discount

        return (
            obs,
            action,
            reward,
            discount,
            next_obs,
            s_a,
            ns_a,
            odom,
            nodom,
        )

    def __iter__(self):
        while True:
            yield self._sample()

    def convert_coordinates(self, goal_world, ego_world, theta):
        # subtract offset
        goal_ego = goal_world[0] - ego_world[0]
        # rotate counterclockwise by theta (degrees)
        rad = np.deg2rad(theta[0])
        rot_mat = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        new_vec = rot_mat @ goal_ego
        return np.array([new_vec])


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(
    replay_dir,
    max_size,
    batch_size,
    num_workers,
    save_snapshot,
    nstep,
    discount,
    lambda_steer,
    lambda_accel,
    lambda_upright,
    lambda_prox,
    her_ratio=0,
    can=False,
    lambda_lp=1
):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(
        replay_dir,
        max_size_per_worker,
        num_workers,
        nstep,
        discount,
        fetch_every=1000,
        save_snapshot=save_snapshot,
        lambda_steer=lambda_steer,
        lambda_accel=lambda_accel,
        lambda_upright=lambda_upright,
        lambda_prox=lambda_prox,
        her_ratio=her_ratio,
        can=can,
        lambda_lp=lambda_lp
    )

    loader = torch.utils.data.DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    return loader
