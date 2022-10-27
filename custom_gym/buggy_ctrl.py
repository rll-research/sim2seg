import numpy as np
import gym
from gym import spaces
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
import dm_env
from dm_env import specs
from .multi_unity import *
import collections
from typing import Any, NamedTuple
from dm_env import StepType, specs
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import torch
import utils


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

