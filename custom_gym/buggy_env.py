import numpy as np
import gym
from gym import spaces
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
import dm_env
import torch
from dm_env import specs
from typing import Any, NamedTuple
from dm_env import StepType, specs
from .multi_unity import *
import collections
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    achieved_goal: Any
    desired_goal: Any
    odom: Any
    rot: Any
    info: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)

    def render(self):
        return self._env.render()

class TrajTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    achieved_goal: Any
    desired_goal: Any
    odom: Any
    rot: Any
    info: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)

    def render(self):
        return self._env.render()

class BuggyGCEnv(gym.GoalEnv):
    """Custom Environment that follows gym interface"""

    def __init__(self, unity_env_name, base_port=5005, num_agents=0, num_odom=10, worker_id=0, rot=False, seed=0, seq_goal=True, final_eval=False):
        super(BuggyGCEnv, self).__init__()

        engine_channel = EngineConfigurationChannel()
        engine_channel.set_configuration_parameters(time_scale=8.0)
        unity_env = UnityToGymWrapper(UnityEnvironment(
                                        file_name=unity_env_name, 
                                        base_port=base_port, 
                                        worker_id=base_port,
                                        seed=seed,
                                        side_channels=[engine_channel]
                                      ),
                                      uint8_visual=False,
                                      flatten_branched=False,
                                      allow_multiple_obs=True,
                                      final_eval=final_eval)
        self.unity_env = unity_env
        self.num_odom = num_odom
        self.rot = rot
        self.num_agents = num_agents
        self.seq_goal = seq_goal
        self.final_eval = final_eval

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        observation = spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.float32)
        desired_goal = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )
        achieved_goal = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )
        odom = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_odom, 2), dtype=np.float32
        )
        rot = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2*int(self.rot),), dtype=np.float32
        )

        observation_space = {
            "desired_goal": desired_goal,
            "achieved_goal": achieved_goal,
            "odom": odom,
        }

        self.observation_space = spaces.Dict(observation_space)

    def step(self, action):
        unity_obs, unity_reward, unity_done, unity_info = self.unity_env.step(action)
        obs = self._get_obs_from_unity(unity_obs, unity_info)
        return obs, unity_reward, unity_done, unity_info

    def reset(self):
        unity_obs = self.unity_env.reset()
        return self._get_obs_from_unity(unity_obs)

    def render(self, mode="human"):
        out = self.unity_env.render()
        out = (out * 255).astype(np.uint8)
        return out

    def close(self):
        self.unity_env.close()

    def _get_obs_from_unity(self, unity_obs, info=None):
        obs = {}
        desired_goal = {}
        achieved_goal = {}

        i = 0
        extra_obs = None
        for obs in unity_obs:
            if obs.shape[-1] != 3:
                vecobs = obs
            elif obs.shape[-1] == 3:
                if i == 0 and self.final_eval:
                    extra_obs = obs
                else:
                    pixobs = obs 
                i += 1

        achieved_goal = vecobs[:2]
        desired_goal = vecobs[2:4]

        odom = vecobs[4 : 4 + (self.num_odom * 3)]
        odom = odom.reshape((self.num_odom, 3))

        if self.seq_goal:
            goalId = vecobs[4 + self.num_odom * 3]
            completedGoals = vecobs[4 + self.num_odom * 3 + 1]
            rot = vecobs[4 + self.num_odom * 3 + 2: 4 + self.num_odom * 3 + 3 + 2]
        elif self.rot:
            rot = vecobs[4 + self.num_odom * 3: 4 + self.num_odom * 3 + 3]
        if info is not None:
            info["collided"] = vecobs[-1]
        return {
            "pixels": np.expand_dims(pixobs, axis=0),
            "desired_goal": np.expand_dims(desired_goal, axis=0),
            "achieved_goal": np.expand_dims(achieved_goal, axis=0),
            "odom": np.expand_dims(odom, axis=0),
            "rot": np.expand_dims(rot, axis=0),
            "goalId": np.expand_dims(goalId, axis=0),
            "completedGoals": np.expand_dims(completedGoals, axis=0), 
            "extra_obs": extra_obs
        }

class FullBuggyWrapper(dm_env.Environment):
    """
    Custom Environment that follows gym interface
    - DMWrapper
    - action adjustment
    - no action repeat
    """

    def __init__(self, env, can, s2s, final_eval, n_actions):
        super(FullBuggyWrapper, self).__init__()
        self._env = env
        self.num_agents = self._env.num_agents
        self.n_actions = n_actions
        self.can = can
        self.s2s = s2s
        self.final_eval = final_eval
        BLACK = np.array([0, 0, 0]) # sky
        GREEN = np.array([0, 128, 1]) # tree / bushes / details
        BLUE = np.array([18, 21, 151]) # ground
        WHITE = np.array([255, 255, 255]) # road
        RED = np.array([215, 10, 42]) # rock
        PURPLE = np.array([214, 47, 251]) # logs

        # for visualizing segmentation maps
        COLORS_LST_DECODE = [BLACK, GREEN, BLUE, WHITE, RED, PURPLE]
        self.COLORS_NP_DECODE = np.array(COLORS_LST_DECODE)

    def step(self, action):
        if self._reset_next_step:
            return self.reset()

        # transform action
        action = action.astype(np.float32)

        observation, rew, done, _ = self._env.step(action)

        if done:
          self._reset_next_step = True
          time_step = dm_env.TimeStep(
              dm_env.StepType.LAST, rew, done, observation)
        else:
          time_step = dm_env.TimeStep(dm_env.StepType.MID, rew, done, observation)
        return self._augment_time_step(time_step, action)

    def reset(self):
        self._reset_next_step = False
        self._step_count = 0

        observation = self._env.reset()

        time_step = dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=np.zeros((self._env.num_agents), dtype=np.float32),
            discount=np.ones((self._env.num_agents), dtype=np.float32),
            observation=observation)
        return self._augment_time_step(time_step)

    def render(self, mode="human"):
        gt = self._env.render()
        if not self.can:
            return gt 
        else:
            if self.final_eval:
                logits = self.__s2s(gt[:, :256, :]/255)
            else:
                logits = self.__s2s(gt/255)
            seg = self.COLORS_NP_DECODE[np.argmax(logits, axis=0)]
            out = np.concatenate((gt, seg), axis=1).astype(np.uint8)
            return out

    def close(self):
        self._env.close()

    def action_spec(self):
        return specs.BoundedArray((1, self.n_actions, 2), np.float32, minimum=-1.0, maximum=1.0, name="action")

    def observation_spec(self):
        if self.can == False:
            result = specs.BoundedArray(
                shape=(1, 3, 256, 256), dtype=np.float32, minimum=0, maximum=255, name="observation"
            )
        else:
            result = specs.BoundedArray(
                shape=(1, 1, 256, 256), dtype=np.uint8, minimum=0, maximum=255, name="observation"
            )
        return result

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        obs = self.__s2s(time_step.observation["pixels"])
        return ExtendedTimeStep(
            observation=obs,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward,
            discount=1.0,
            desired_goal=time_step.observation["desired_goal"],
            achieved_goal=time_step.observation["achieved_goal"],
            odom=time_step.observation["odom"],
            rot=time_step.observation["rot"],
            info={'goalId': time_step.observation["goalId"],
                  'completedGoals': time_step.observation['completedGoals']}
        )

    def __s2s(self, pixels):
        if not self.can:
            if len(pixels.shape) == 5:
                pixels = pixels.reshape((-1, pixels.shape[2], pixels.shape[3], pixels.shape[4]))
            if len(pixels.shape) == 4:
                pixelsT = pixels.transpose(0, 3, 1, 2).copy()
            else:
                pixelsT = pixels.transpose(2, 0, 1).copy()
            return pixelsT

        if len(pixels.shape) == 5:
            pixelsT = pixels.reshape(-1, pixels.shape[2], pixels.shape[3], pixels.shape[4])
            pixelsT = torch.from_numpy(pixelsT.transpose(0, 3, 1, 2).copy()).to(
                self.s2s.device
            )
        elif len(pixels.shape) == 4:
            pixelsT = torch.from_numpy(pixels.transpose(0, 3, 1, 2).copy()).to(
                self.s2s.device
            )
        else:
            pixelsT = (
                torch.from_numpy(pixels.transpose(2, 0, 1).copy())
                .unsqueeze(0)
                .to(self.s2s.device)
            )
        pixelsT = pixelsT.float()

        noise = torch.zeros((1, 3, 256, 256)).to(self.s2s.device)
        self.s2s.set_input(
            {"A": pixelsT, "B": noise, "A_paths": noise, "B_paths": noise, "D": noise}
        )
        self.s2s.test()
        with torch.no_grad():
            self.s2s.forward()

        if len(pixels.shape) == 4:
            out = self.s2s.fake_B.cpu().numpy()
        elif len(pixels.shape) == 5:
            out = self.s2s.fake_B.cpu().numpy()
        else:
            out = self.s2s.fake_B.squeeze(0).cpu().numpy()
        
        return out

class LSTMBuggyWrapper(dm_env.Environment):
    def __init__(self, env, agent, num_actions=1):
        super(LSTMBuggyWrapper, self).__init__()
        self._env = env
        self.prev_ts = None 
        self.max_iters = 200
        self.prev_frames = []
        self.agent = agent
        if self.agent:
            self.num_actions = self.agent.num_actions 
        else:
            self.num_actions = num_actions

    def step(self, action):
        if self._reset_next_step:
            return self.reset()

        reward = 0
        for i in range(action.shape[-2]):
            if self.prev_ts.last():
                self._reset_next_step = True
                return self._augment_time_step(self.prev_ts, action)

            # take environment step
            sub_action = action[:, i, :]
            self.prev_ts = self._env.step(sub_action)
            reward += self.prev_ts.reward

            # possibly end episode
            self.prev_frames.append(self._env.render())

        info = {}
        return self._augment_time_step(self.prev_ts, action, info=info, reward=reward)

    def reset(self):
        self._reset_next_step = False
        self._step_count = 0

        ts = self._env.reset()
        time_step = ExtendedTimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=np.zeros((self._env.num_agents), dtype=np.float32),
            discount=np.ones((self._env.num_agents), dtype=np.float32),
            observation=ts.observation,
            desired_goal=ts.desired_goal,
            achieved_goal=ts.achieved_goal,
            odom=ts.odom,
            rot=ts.rot,
            action=None,
            info={'goalId': 0,
                  'completedGoals': 0})
        self.prev_ts = self._augment_time_step(time_step)
        self.prev_frames = []
        self.prev_frames.append(self._env.render())
        return self.prev_ts

    def render(self, mode="human"):
        frames = self.prev_frames
        self.prev_frames = []
        return frames

    def close(self):
        self._env.close()

    def action_spec(self):
        """Returns the action specification for this environment."""
        return self._env.action_spec()


    def observation_spec(self):
        return self._env.observation_spec()

    def _augment_time_step(self, time_step, action=None, info=None, reward=0):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        if reward is None:
            reward = time_step.reward
        if time_step.info is not None:
            if info is not None:
                time_step.info.update(info)
                combined_info = time_step.info
            else:
                combined_info = time_step.info
        elif info is not None:
            combined_info = info
        return TrajTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=reward,
            discount=1.0,
            desired_goal=time_step.desired_goal,
            achieved_goal=time_step.achieved_goal,
            odom=time_step.odom,
            rot=time_step.rot,
            info=combined_info
        )
