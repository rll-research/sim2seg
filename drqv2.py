# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import utils

from options.base_options_simple import BaseOptionsSimple
import models
from models import create_model

NUM_CLASSES = 6

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shapes, freeze=True):
        super().__init__()

        assert len(obs_shapes[0]) == 3
        obs_shape = obs_shapes[0]
        self.repr_dim = 468544

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )

        self.apply(utils.weight_init)

    def forward(self, img, s_a, odom=None):
        img = img / 255.0 - 0.5
        h = self.convnet(img)
        h = h.view(h.shape[0], -1)

        return torch.cat(
            (h, s_a)
            if odom is None
            else (h, s_a, odom.view(odom.shape[0], -1)),
            dim=1,
        )


class SegEncoder(nn.Module):
    def __init__(self, obs_shapes, freeze=True):
        super().__init__()

        assert len(obs_shapes[0]) == 3
        obs_shape = obs_shapes[0]
        self.repr_dim = 468544  # 32 * 35 * 35 + 24

        self.convnet = nn.Sequential(
            nn.Conv2d(NUM_CLASSES, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )

        self.apply(utils.weight_init)

    def forward(self, img, s_a, odom=None):
        img = img / 255.0 - 0.5
        h = self.convnet(img)
        h = h.view(h.shape[0], -1)

        return torch.cat(
            (h, s_a)
            if odom is None
            else (h, s_a, odom.view(odom.shape[0], -1)),
            dim=1,
        )

def extract_modules_from_seq(net, modules):
    layers = net.model
    assert isinstance(layers, nn.Sequential), "must call on nn Sequential"
    for layer in layers:
        if (isinstance(layer, models.networks.UnetSkipConnectionBlock)):
            extract_modules_from_seq(layer, modules)
            return
        modules += [layer]


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0]),
        )

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist

class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, num_actions):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )
        self.num_actions = num_actions

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0]*self.num_actions, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0]*self.num_actions, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2

class LSTMActor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, num_actions):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )

        self.num_layers = 1
        self.H_out = action_shape[0]
        self.H_cell = hidden_dim
        self.policy = nn.LSTM(feature_dim, self.H_cell,
                            num_layers=self.num_layers, batch_first=True)
        self.H_out = nn.Sequential(nn.Linear(self.H_cell, self.H_cell // 2),
                            nn.ReLU(inplace=True),
                            nn.Linear(self.H_cell // 2, action_shape[0]))
        self.num_actions = num_actions

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        obs_f = self.trunk(obs)
        h = torch.zeros((self.num_layers, obs.shape[0], self.H_cell)).to(obs.device)
        c = torch.zeros((self.num_layers, obs.shape[0], self.H_cell)).to(obs.device)

        obs_f = obs_f.unsqueeze(1)
        obs_f = torch.tile(obs_f, (1, self.num_actions, 1))
        mu, _  = self.policy(obs_f, (h, c))
        b = mu.shape[0]
        mu = mu.reshape((-1, mu.shape[-1]))
        mu = self.H_out(mu)
        mu = mu.reshape((b, -1, mu.shape[-1]))

        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist

class DrQV2LSTMAgent:
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        feature_dim,
        hidden_dim,
        lstm_hidden_dim,
        critic_target_tau,
        num_expl_steps,
        update_every_steps,
        stddev_schedule,
        stddev_clip,
        use_tb,
        num_actions,
        use_s2s
    ):

        print(f"initializing agent on {device}")
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        self.num_actions = num_actions

        # models
        enc = SegEncoder if use_s2s else Encoder
        self.encoder = enc(obs_shape).to(device)
        self.actor = LSTMActor(
            self.encoder.repr_dim, action_shape, feature_dim, lstm_hidden_dim, num_actions,
        ).to(device)

        self.critic = Critic(
            self.encoder.repr_dim, action_shape, feature_dim, hidden_dim, num_actions
        ).to(device)
        self.critic_target = Critic(
            self.encoder.repr_dim, action_shape, feature_dim, hidden_dim, num_actions
        ).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        # if not self.bottleneck:
        #     self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, s_a, odom, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device) # not sure why this is weird
        if len(obs.shape) == 3:
            obs = self.encoder(
                obs.unsqueeze(0), s_a.unsqueeze(0), odom.unsqueeze(0)
            )
        else:
            obs = self.encoder(obs, s_a, odom)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()

    def get_traj(self, obs, s_g, odom, dt=0.1):
        """
        prediction is in unity space, will be converted in agent
        input: obs --> (1 x 3 x 256 x 256)
        input: odom data --> (1 x 10 x 3)
        input: s_g --> (2,)
        output: proposed trajectory (3d) --> (1 x 10 x 3)
        """

        with torch.no_grad():
            action = self.act(
                obs,
                s_g,
                odom,
                1,
                eval_mode=True,
            ).squeeze()
            # angle: (-1.0, 1.0) --> (-pi/2, pi/2)
            # accel: (-1.0, 1.0) --> (max_accel, max_accel)

        delta = (odom[0][-1] - odom[0][-2]).cpu()

        # (right, up, forward) --> (right, forward)
        delta = np.array([delta[0], delta[2]])
        speed = np.linalg.norm(delta)
        heading = 0
        curr_angle = 0
        out_traj = np.zeros((1, 10, 3))
        idx = 0

        for act_ind in range(self.num_actions):
            if len(action.shape) == 1:
                angle = np.pi / 4 * action[0]
                accel = 2 * action[1]
            else:
                angle = np.pi / 4 * action[act_ind][0]
                accel = 2 * action[act_ind][1]
            for i in range(10//self.num_actions):
                if idx == 9:
                    break
                # idea: heading is the vehicle direction in own frame
                # angle is the target steer that the model requests
                # curr_angle is the current angle of the steer; update this at each step to approach angle
                # update heading at every step using curr_angle
                speed += accel * dt
                curr_angle = np.clip(
                    curr_angle + 0.05 * np.sign(angle) * np.pi / 4,
                    -abs(angle),
                    abs(angle),
                )
                speed = np.clip(speed, 0, 5)
                val = np.array([0, speed])
                if speed < 0:
                    heading += curr_angle
                else:
                    heading -= curr_angle
                c, s = np.cos(heading), np.sin(heading)
                R = np.array(((c, -s), (s, c)))
                res = R @ val
                out_traj[0, (idx + 1)] = out_traj[
                    0, (idx)
                ] + np.array([res[0], 0, res[1]])
                idx += 1

        return out_traj

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()
        action = action.reshape((action.shape[0], -1))

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            next_action = next_action.reshape((next_action.shape[0], -1))
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_q1"] = Q1.mean().item()
            metrics["critic_q2"] = Q2.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        action = action.reshape((action.shape[0], -1))
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_logprob"] = log_prob.mean().item()
            metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        (
            obs,
            action,
            reward,
            discount,
            next_obs,
            s_a,
            ns_a,
            odom,
            nodom,
        ) = utils.replay_to_torch(batch, self.device)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs, s_a, odom)
        with torch.no_grad():
            next_obs = self.encoder(next_obs, ns_a, nodom)

        if self.use_tb:
            metrics["batch_reward"] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step)
        )

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics
