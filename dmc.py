# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs

from custom_gym.buggy_env import *
import os
import torch

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


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += time_step.reward * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def render(self):
        return self._env.render()

    def __getattr__(self, name):
        return getattr(self._env, name)

    def close(self):
        self._env.close()

_ACTION_SPEC_MUST_BE_BOUNDED_ARRAY = (
    "`env.action_spec()` must return a single `BoundedArray`, got: {}.")
_MUST_BE_FINITE = "All values in `{name}` must be finite, got: {bounds}."
_MUST_BROADCAST = (
    "`{name}` must be broadcastable to shape {shape}, got: {bounds}.")


class ActionScaleWrapper(dm_env.Environment):
  """Wraps a control environment to rescale actions to a specific range."""
  __slots__ = ("_action_spec", "_env", "_transform")

  def __init__(self, env, minimum, maximum):
    """Initializes a new action scale Wrapper.
    Args:
      env: Instance of `dm_env.Environment` to wrap. Its `action_spec` must
        consist of a single `BoundedArray` with all-finite bounds.
      minimum: Scalar or array-like specifying element-wise lower bounds
        (inclusive) for the `action_spec` of the wrapped environment. Must be
        finite and broadcastable to the shape of the `action_spec`.
      maximum: Scalar or array-like specifying element-wise upper bounds
        (inclusive) for the `action_spec` of the wrapped environment. Must be
        finite and broadcastable to the shape of the `action_spec`.
    Raises:
      ValueError: If `env.action_spec()` is not a single `BoundedArray`.
      ValueError: If `env.action_spec()` has non-finite bounds.
      ValueError: If `minimum` or `maximum` contain non-finite values.
      ValueError: If `minimum` or `maximum` are not broadcastable to
        `env.action_spec().shape`.
    """
    action_spec = env.action_spec()
    if not isinstance(action_spec, specs.BoundedArray):
      raise ValueError(_ACTION_SPEC_MUST_BE_BOUNDED_ARRAY.format(action_spec))

    minimum = np.array(minimum)
    maximum = np.array(maximum)
    shape = action_spec.shape
    orig_minimum = np.array([-1, 0])
    orig_maximum = np.array([1, 0.5]) #action_spec.maximum
    orig_dtype = action_spec.dtype

    def validate(bounds, name):
      if not np.all(np.isfinite(bounds)):
        raise ValueError(_MUST_BE_FINITE.format(name=name, bounds=bounds))
      try:
        np.broadcast_to(bounds, shape)
      except ValueError:
        raise ValueError(_MUST_BROADCAST.format(
            name=name, bounds=bounds, shape=shape))

    validate(minimum, "minimum")
    validate(maximum, "maximum")
    validate(orig_minimum, "env.action_spec().minimum")
    validate(orig_maximum, "env.action_spec().maximum")

    scale = (orig_maximum - orig_minimum) / (maximum - minimum)

    def transform(action):
      new_action = orig_minimum + scale * (action - minimum)
      return new_action.astype(orig_dtype, copy=False)

    dtype = np.result_type(minimum, maximum, orig_dtype)
    self._action_spec = action_spec.replace(
        minimum=minimum, maximum=maximum, dtype=dtype)
    self._env = env
    self._transform = transform

  def step(self, action):
    return self._env.step(self._transform(action))

  def reset(self):
    return self._env.reset()

  def observation_spec(self):
    return self._env.observation_spec()

  def action_spec(self):
    return self._action_spec

  def __getattr__(self, name):
    return getattr(self._env, name)

class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            dtype,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        out =  self._env.step(action)
        return out

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def render(self):
        return self._env.render()

    def __getattr__(self, name):
        return getattr(self._env, name)

    def close(self):
        self._env.close()


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def render(self):
        return self._env.render()

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation["pixels"],
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward,
            discount=1.0,
            desired_goal=np.expand_dims(time_step.observation["desired_goal"], axis=0),
            achieved_goal=np.expand_dims(time_step.observation["achieved_goal"], axis=0),
            odom=time_step.observation["odom"],
            rot=time_step.observation["rot"],
            info={}
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

    def close(self):
        self._env.close()    

def unity_lstm_make(name, frame_stack, action_repeat, seed, base_port, can=True, num_odom=10, rot=False, all_envs=False, s2s=None, agent=None, device=None, num_actions=1, seq_goal=True, final_eval=False):
    env_fn = BuggyGCEnv #BuggyCtrlEnv
    unity_kwargs = {"base_port": base_port, "num_odom": num_odom, "rot": rot, "seed": seed, "seq_goal": seq_goal, "final_eval": final_eval}
    env = env_fn(name, **unity_kwargs)
    env = FullBuggyWrapper(env, can, s2s, final_eval, num_actions) 
    env = LSTMBuggyWrapper(env, agent, num_actions)
    return env
