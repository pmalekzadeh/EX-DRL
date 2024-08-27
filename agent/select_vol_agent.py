"""Generic actor implementation, using TensorFlow and Sonnet."""

from typing import Optional, Tuple

from acme import adders
from acme import core
from acme import types
# Internal imports.

import dm_env
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

import copy
import dataclasses
import operator
import functools
from typing import Iterator, List, Optional, Tuple

from acme import datasets
from acme import specs
from acme.adders import reverb as reverb_adders
from acme.adders.reverb import base as reverb_base
from acme.agents import agent
from acme.agents.tf import actors
from acme.tf import networks as network_utils
from acme.tf import utils
from acme.tf import variable_utils
from acme.utils import counting
from acme.utils import loggers

import reverb
import sonnet as snt
import tensorflow as tf
import agent.learning as learning
import tree

import numpy as np
from .agent import D4PGConfig

tfd = tfp.distributions

class SelectVolFeedForwardActor(core.Actor):
  """A feed-forward actor.

  An actor based on a feed-forward policy which takes non-batched observations
  and outputs non-batched actions. It also allows adding experiences to replay
  and updating the weights from the policy on the learner.
  """

  def __init__(
      self,
      policy_networks: {float:snt.Module},
      adder: Optional[adders.Adder] = None,
      variable_client: Optional[variable_utils.VariableClient] = None,
  ):
    """Initializes the actor.

    Args:
      policy_network: the policy to run.
      adder: the adder object to which allows to add experiences to a
        dataset/replay buffer.
      variable_client: object which allows to copy weights from the learner copy
        of the policy to the actor copy (in case they are separate).
    """

    # Store these for later use.
    self._adder = adder
    self._variable_client = variable_client
    self._policy_networks = policy_networks
    
  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    # Pass the observation through the policy network.
    # select trained policy based on vol
    
    vol = np.round(observation[1], decimals=1)
    vol = 0.8 if vol > 0.8 else vol
    vol = str(0.1) if vol < 0.1 else str(vol)
      
    # Add a dummy batch dimension and as a side effect convert numpy to TF.
    batched_observation = utils.add_batch_dim(observation)

    # Compute the policy, conditioned on the observation.
    policy = self._policy_networks[vol](batched_observation)

    # Sample from the policy if it is stochastic.
    action = policy.sample() if isinstance(policy, tfd.Distribution) else policy
    
    # Return a numpy array with squeezed out batch dimension.
    return utils.to_numpy_squeeze(action)

  def observe_first(self, timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add_first(timestep)

  def observe(self, action: types.NestedArray, next_timestep: dm_env.TimeStep):
    if self._adder:
      self._adder.add(action, next_timestep)

  def update(self, wait: bool = False):
    if self._variable_client:
      self._variable_client.update(wait)


