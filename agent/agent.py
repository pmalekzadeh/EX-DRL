# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""D4PG agent implementation."""

import copy
import dataclasses
import operator
import functools
from typing import Iterator, List, Optional, Tuple

from acme import adders
from acme import core
from acme import datasets
from acme import specs
from acme import types
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

def _n_step_transition_from_episode(observations: types.NestedTensor,
                                    actions: tf.Tensor,
                                    rewards: tf.Tensor,
                                    discounts: tf.Tensor,
                                    n_step: int,
                                    discount: float):
    """Produce Reverb-like N-step transition from a full episode.

    Observations, actions, rewards and discounts have the same length. This
    function will ignore the first reward and discount and the last action.

    Args:
        observations: [L, ...] Tensor.
        actions: [L, ...] Tensor.
        rewards: [L] Tensor.
        discounts: [L] Tensor.
        n_step: number of steps to squash into a single transition.
        discount: discount to use for TD updates.

    Returns:
        (o_t, a_t, r_t, d_t, o_tp1) tuple.
    """

    max_index = tf.shape(rewards)[0] - 1
    first = tf.random.uniform(shape=(), minval=0, maxval=max_index - 1,
                              dtype=tf.int32)
    last = tf.minimum(first + n_step, max_index)

    o_t = tree.map_structure(operator.itemgetter(first), observations)
    a_t = tree.map_structure(operator.itemgetter(first), actions)
    o_tp1 = tree.map_structure(operator.itemgetter(last), observations)

    # 0, 1, ..., n-1.
    discount_range = tf.cast(tf.range(last - first), tf.float32)
    # 1, g, ..., g^{n-1}.
    additional_discounts = tf.pow(discount, discount_range)
    # 1, d_t, d_t * d_{t+1}, ..., d_t * ... * d_{t+n-2}.
    discounts = tf.concat([[1.], tf.math.cumprod(discounts[first:last-1])], 0)
    # 1, g * d_t, ..., g^{n-1} * d_t * ... * d_{t+n-2}.
    discounts *= additional_discounts
    #  r_t + g * d_t * r_{t+1} + ... + g^{n-1} * d_t * ... * d_{t+n-2} * r_{t+n-1}
    # We have to shift rewards by one so last=max_index corresponds to transitions
    # that include the last reward.
    r_t = tf.reduce_sum(rewards[first+1:last+1] * discounts)

    # g^{n-1} * d_{t} * ... * d_{t+n-1}.
    d_t = discounts[-1]

    info = tree.map_structure(lambda dtype: tf.ones([], dtype),
                              reverb.SampleInfo.tf_dtypes())
    return reverb.ReplaySample(
        info=info, data=types.Transition(o_t, a_t, r_t, d_t, o_tp1))


############################################################ D4PG with GPD for tail modelling

@dataclasses.dataclass
class GPDConfig:
    """Configuration options for the GPD agent."""
    obj_func: str = 'var'
    critic_loss_type: str = 'gpd'
    threshold: float = 0.95
    GPD_threshold: float = 0.96, ####
    n_GPD_samples: int =48,  # num_samples in the GPD tail
    quantile_interval:float =0.01,
    discount: float = 0.99
    batch_size: int = 256
    prefetch_size: int = 4
    target_update_period: int = 100
    policy_optimizer: Optional[snt.Optimizer] = None
    critic_optimizer: Optional[snt.Optimizer] = None
    GPD_optimizer: Optional[snt.Optimizer] = None
    min_replay_size: int = 1000
    max_replay_size: int = 1000000
    samples_per_insert: Optional[float] = 32.0
    n_step: int = 20
    sigma: float = 0.3
    demo_step: int = 1000
    clipping: bool = True
    per: bool = False
    priority_exponent: float = 0.6
    importance_sampling_exponent: float = 0.2
    replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE


@dataclasses.dataclass
class GPDNetworks:
    """Structure containing the networks for GPD."""
    scale_network: snt.Module
    shape_network: snt.Module
    observation_network: snt.Module
    policy_network: snt.Module
    critic_network: snt.Module

    def __init__(
        self,
        scale_network: snt.Module,
        shape_network: snt.Module,
        policy_network: snt.Module,
        critic_network: snt.Module,
        observation_network: types.TensorTransformation,
    ):
        # This method is implemented (rather than added by the dataclass decorator)
        # in order to allow observation network to be passed as an arbitrary tensor
        # transformation rather than as a snt Module.
        # TODO(mwhoffman): use Protocol rather than Module/TensorTransformation.
        self.scale_network = scale_network
        self.shape_network = shape_network
        self.policy_network = policy_network
        self.critic_network = critic_network
        self.observation_network = utils.to_sonnet_module(observation_network)

    def init(self, environment_spec: specs.EnvironmentSpec):
        """Initialize the networks given an environment spec."""
        # Get observation and action specs.
        act_spec = environment_spec.actions
        obs_spec = environment_spec.observations

        # Create variables for the observation net and, as a side-effect, get a
        # spec describing the embedding space.
        emb_spec = utils.create_variables(self.observation_network, [obs_spec])

        # Create variables for the scale and shape nets.
        _ = utils.create_variables(self.scale_network, [emb_spec, act_spec])
        _ = utils.create_variables(self.shape_network, [emb_spec, act_spec])
        _ = utils.create_variables(self.policy_network, [emb_spec])
        _ = utils.create_variables(self.critic_network, [emb_spec, act_spec])

    def make_policy(
        self,
        environment_spec: specs.EnvironmentSpec,
        sigma: float = 0.0,
    ) -> snt.Module:
        """Create a single network which evaluates the policy."""
        # Stack the observation and policy networks.
        stack = [
            self.observation_network,
            self.policy_network,
        ]

        # If a stochastic/non-greedy policy is requested, add Gaussian noise on
        # top to enable a simple form of exploration.
        # TODO(mwhoffman): Refactor this to remove it from the class.
        if sigma > 0.0:
            stack += [
                network_utils.ClippedGaussian(sigma),
                network_utils.ClipToSpec(environment_spec.actions),
            ]

        # Return a network which sequentially evaluates everything in the stack.
        return snt.Sequential(stack)


        # Return a network which sequentially evaluates everything in the stack.
        return snt.Sequential(stack)


class GPDBuilder:
    """Builder for GPD which constructs individual components of the agent."""

    def __init__(self, config: GPDConfig):
        self._config = config

    def make_replay_tables(
        self,
        environment_spec: specs.EnvironmentSpec,
    ) -> List[reverb.Table]:
        """Create tables to insert data into."""
        if self._config.samples_per_insert is None:
            # We will take a samples_per_insert ratio of None to mean that there is
            # no limit, i.e. this only implies a min size limit.
            limiter = reverb.rate_limiters.MinSize(
                self._config.min_replay_size)
        else:
            # Create enough of an error buffer to give a 10% tolerance in rate.
            samples_per_insert_tolerance = 0.1 * self._config.samples_per_insert
            error_buffer = self._config.min_replay_size * samples_per_insert_tolerance
            limiter = reverb.rate_limiters.SampleToInsertRatio(
                min_size_to_sample=self._config.min_replay_size,
                samples_per_insert=self._config.samples_per_insert,
                error_buffer=error_buffer)
        replay_table = reverb.Table(
            name=self._config.replay_table_name,
            sampler=reverb.selectors.Uniform() if not self._config.per
                else reverb.selectors.Prioritized(priority_exponent=self._config.priority_exponent),
            remover=reverb.selectors.Fifo(),
            max_size=self._config.max_replay_size,
            rate_limiter=limiter,
            signature=reverb_adders.NStepTransitionAdder.signature(
                environment_spec))
        return [replay_table]

    def make_dataset_iterator(
        self,
        reverb_client: reverb.Client,
        demonstration_dataset: tf.data.Dataset = None,
    ) -> Tuple[Iterator[reverb.ReplaySample]]:
        """Create a dataset iterator to use for learning/updating the agent."""
        # The dataset provides an interface to sample from replay.
        dataset = datasets.make_reverb_dataset(
            table=self._config.replay_table_name,
            server_address=reverb_client.server_address,
            batch_size=self._config.batch_size,
            prefetch_size=self._config.prefetch_size)

        if demonstration_dataset is not None:
            transition = functools.partial(_n_step_transition_from_episode,
                                           n_step=self._config.n_step,
                                           discount=self._config.discount)
            dataset_demos = demonstration_dataset.map(transition)
            # Batch and prefetch
            dataset_demos = dataset_demos.batch(
                self._config.batch_size, drop_remainder=True)
            dataset_demos = dataset_demos.prefetch(self._config.prefetch_size)
            dataset_demos_iter = iter(dataset_demos)
        else:
            dataset_demos_iter = None

        return iter(dataset), dataset_demos_iter

    def make_adder(
        self,
        replay_client: reverb.Client,
    ) -> adders.Adder:
        """Create an adder which records data generated by the actor/environment."""
        return reverb_adders.NStepTransitionAdder(
            priority_fns={self._config.replay_table_name: lambda x: 1.},
            client=replay_client,
            n_step=self._config.n_step,
            discount=self._config.discount)

    def make_actor(
        self,
        policy_network: snt.Module,
        adder: Optional[adders.Adder] = None,
        variable_source: Optional[core.VariableSource] = None,
    ):
        """Create an actor instance."""
        if variable_source:
            # Create the variable client responsible for keeping the actor up-to-date.
            variable_client = variable_utils.VariableClient(
                client=variable_source,
                variables={'policy': policy_network.variables},
                update_period=1000,
            )
            # Make sure not to use a random policy after checkpoint restoration by
            # assigning variables before running the environment loop.
            variable_client.update_and_wait()

        else:
            variable_client = None
        # Create the actor which defines how we take actions.
        return actors.FeedForwardActor(
            policy_network=policy_network,
            adder=adder,
            variable_client=variable_client,
        )

    def make_learner(
        self,
        networks: Tuple[GPDNetworks, GPDNetworks],
        dataset: Iterator[reverb.ReplaySample],
        demo_dataset: Iterator[reverb.ReplaySample] = None,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        checkpoint: bool = False,
        checkpoint_folder: str = '~/acme',
        replay_client: reverb.Client = None,
    ):
        """Creates an instance of the learner."""
        online_networks, target_networks = networks


        # The learner updates the parameters (and initializes them).
        return learning.GPDLearner(
            obj_func=self._config.obj_func,
            threshold=self._config.threshold,
            GPD_threshold= self._config.GPD_threshold,
            n_GPD_samples=self._config.n_GPD_samples,
            quantile_interval=self._config.quantile_interval,
            policy_network=online_networks.policy_network,
            critic_network=online_networks.critic_network,
            observation_network=online_networks.observation_network,
            target_policy_network=target_networks.policy_network,
            target_critic_network=target_networks.critic_network,
            target_observation_network=target_networks.observation_network,
            ###### online and target GPD networks
            GPD_scale_network=online_networks.scale_network,
            GPD_shape_network=online_networks.shape_network,
            target_GPD_scale_network=target_networks.scale_network,
            target_GPD_shape_network=target_networks.shape_network,
            policy_optimizer=self._config.policy_optimizer,
            critic_optimizer=self._config.critic_optimizer,
            GPD_optimizer=self._config.GPD_optimizer,
            clipping=self._config.clipping,
            discount=self._config.discount,
            target_update_period=self._config.target_update_period,
            dataset_iterator=dataset,
            demo_dataset_iterator=demo_dataset,
            demo_step=self._config.demo_step,
            per=self._config.per,
            importance_sampling_exponent=self._config.importance_sampling_exponent,
            counter=counter,
            logger=logger,
            checkpoint=checkpoint,
            checkpoint_folder=checkpoint_folder,
            replay_client=replay_client,
        )


class GPD(agent.Agent):
    """GPD Agent.
    This implements a single-process GPD agent. This is an actor-critic algorithm
    that generates data via a behavior policy, inserts N-step transitions into
    a replay buffer, and periodically updates the policy (and as a result the
    behavior) by sampling uniformly from this buffer.
    """

    def __init__(
        self,
        environment_spec: specs.EnvironmentSpec,
        policy_network: snt.Module,
        critic_network: snt.Module,
        scale_network: snt.Module,
        shape_network: snt.Module,
        obj_func='var',
        critic_loss_type='gpd',
        threshold=0.95,
        GPD_threshold=0.96, ###
        n_GPD_samples=48,
        quantile_interval=0.01,
        observation_network: types.TensorTransformation = tf.identity,
        discount: float = 0.99,
        batch_size: int = 256,
        prefetch_size: int = 4,
        target_update_period: int = 100,
        policy_optimizer: Optional[snt.Optimizer] = None,
        critic_optimizer: Optional[snt.Optimizer] = None,
        GPD_optimizer: Optional[snt.Optimizer] = None,
        min_replay_size: int = 1000,
        max_replay_size: int = 1000000,
        samples_per_insert: float = 32.0,
        n_step: int = 5,
        sigma: float = 0.3,
        clipping: bool = True,
        replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE,
        per: bool = False,
        priority_exponent: float = 0.6,
        importance_sampling_exponent: float = 0.2,
        demonstration_step: int = 1_000,
        demonstration_dataset: Optional[tf.data.Dataset] = None,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        checkpoint: bool = True,
        checkpoint_folder: str = '~/acme'
    ):
        """Initialize the agent.
        Args:
          environment_spec: description of the actions, observations, etc.
          policy_network: the online (optimized) policy.
          critic_network: the online critic.
          scale_network: the online GPD's scale.
          shape_network: the online GPD's shape.
          observation_network: optional network to transform the observations before
            they are fed into any network.
          discount: discount to use for TD updates.
          batch_size: batch size for updates.
          prefetch_size: size to prefetch from replay.
          target_update_period: number of learner steps to perform before updating
            the target networks.
          policy_optimizer: optimizer for the policy network updates.
          critic_optimizer: optimizer for the critic network updates.
          min_replay_size: minimum replay size before updating.
          max_replay_size: maximum replay size.
          samples_per_insert: number of samples to take from replay for every insert
            that is made.
          n_step: number of steps to squash into a single transition.
          sigma: standard deviation of zero-mean, Gaussian exploration noise.
          clipping: whether to clip gradients by global norm.
          replay_table_name: string indicating what name to give the replay table.
          counter: counter object used to keep track of steps.
          logger: logger object to be used by learner.
          checkpoint: boolean indicating whether to checkpoint the learner.
        """
        # Create the Builder object which will internally create agent components.
        builder = GPDBuilder(
            # TODO(mwhoffman): pass the config dataclass in directly.
            # TODO(mwhoffman): use the limiter rather than the workaround below.
            # Right now this modifies min_replay_size and samples_per_insert so that
            # they are not controlled by a limiter and are instead handled by the
            # Agent base class (the above TODO directly references this behavior).
            GPDConfig(
                obj_func=obj_func,
                critic_loss_type=critic_loss_type,
                threshold=threshold,
                GPD_threshold=GPD_threshold,
                n_GPD_samples=n_GPD_samples,
                quantile_interval=quantile_interval,
                discount=discount,
                batch_size=batch_size,
                prefetch_size=prefetch_size,
                target_update_period=target_update_period,
                policy_optimizer=policy_optimizer,
                critic_optimizer=critic_optimizer,
                GPD_optimizer=GPD_optimizer,
                min_replay_size=1,  # Let the Agent class handle this.
                max_replay_size=max_replay_size,
                samples_per_insert=None,  # Let the Agent class handle this.
                n_step=n_step,
                sigma=sigma,
                clipping=clipping,
                demo_step=demonstration_step,
                per=per,
                priority_exponent=priority_exponent,
                importance_sampling_exponent=importance_sampling_exponent,
                replay_table_name=replay_table_name if per else 'uniform_fifo',
            ))

        # TODO(mwhoffman): pass the network dataclass in directly.

        online_networks = GPDNetworks(scale_network=scale_network,
                                          shape_network=shape_network,
                                          observation_network=observation_network,
                                          policy_network=policy_network,
                                         critic_network=critic_network,)


        # Target networks are just a copy of the online networks.
        target_networks = copy.deepcopy(online_networks)


        # Initialize the networks.
        online_networks.init(environment_spec)
        target_networks.init(environment_spec)

        # TODO(mwhoffman): either make this Dataclass or pass only one struct.
        # The network struct passed to make_learner is just a tuple for the
        # time-being (for backwards compatibility).
        networks = (online_networks, target_networks)

        # Create the behavior policy.
        policy_network = online_networks.make_policy(environment_spec, sigma)

        # Create the replay server and grab its address.
        replay_tables = builder.make_replay_tables(environment_spec)
        replay_server = reverb.Server(replay_tables, port=None)
        replay_client = reverb.Client(f'localhost:{replay_server.port}')

        # Create actor, dataset, and learner for generating, storing, and consuming
        # data respectively.
        adder = builder.make_adder(replay_client)
        actor = builder.make_actor(policy_network, adder)
        dataset, demo_dataset = builder.make_dataset_iterator(
            replay_client, demonstration_dataset)
        learner = builder.make_learner(networks, dataset, demo_dataset, counter, logger,
                                       checkpoint, checkpoint_folder, replay_client)

        super().__init__(
            actor=actor,
            learner=learner,
            min_observations=max(batch_size, min_replay_size),
            observations_per_step=float(batch_size) / samples_per_insert)

        # Save the replay so we don't garbage collect it.
        self._replay_server = replay_server