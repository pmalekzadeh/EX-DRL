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

"""D4PG learner implementation."""

import time
from typing import Dict, Iterator, List, Optional, Union

import acme
from acme import types
from acme.tf import losses
from acme.tf import networks as acme_nets
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting
from acme.utils import loggers
from acme.adders import reverb as reverb_adders
import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
import tree

from agent.distributional import quantile_regression
from agent.distributional import GPDDistribution
from agent.distributional import GPD_quantile_regression, compute_gpd_loss, quantile_numbers, GPD_samples

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

################ GPDLearner
class GPDLearner(acme.Learner):
    """GPD learner.
    This is the learning component of a GPD agent. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        policy_network: snt.Module,
        critic_network: snt.Module,
        target_policy_network: snt.Module,
        target_critic_network: snt.Module,
        GPD_scale_network: snt.Module,
        GPD_shape_network: snt.Module,
        target_GPD_scale_network: snt.Module,
        target_GPD_shape_network: snt.Module,
        discount: float,
        target_update_period: int,
        dataset_iterator: Iterator[reverb.ReplaySample],
        demo_dataset_iterator: Optional[Iterator[reverb.ReplaySample]] = None,
        demo_step: Optional[int] = 1_000,
        obj_func='var',
        threshold=0.95,
        GPD_threshold=0.96,
        n_GPD_samples= 48,
        quantile_interval=0.01,
        observation_network: types.TensorTransformation = lambda x: x,
        target_observation_network: types.TensorTransformation = lambda x: x,
        policy_optimizer: Optional[snt.Optimizer] = None,
        critic_optimizer: Optional[snt.Optimizer] = None,
        GPD_optimizer: Optional[snt.Optimizer] = None,
        clipping: bool = True,
        per: bool = False,
        importance_sampling_exponent: float = 0.2,
        counter: Optional[counting.Counter] = None,
        logger: Optional[loggers.Logger] = None,
        checkpoint: bool = True,
        checkpoint_folder: str = '',
        replay_client: Optional[Union[reverb.Client, reverb.TFClient]] = None,
    ):
        """Initializes the learner.
        Args:
          policy_network: the online (optimized) policy.
          critic_network: the online critic.
          target_policy_network: the target policy (which lags behind the online
            policy).
          target_critic_network: the target critic.
          discount: discount to use for TD updates.
          target_update_period: number of learner steps to perform before updating
            the target networks.
          obj_func: objective function for policy gradient update. (var or cvar)
          critic_loss_type: c51 or qr.
          threshold: threshold for objective function
          dataset_iterator: dataset to learn from, whether fixed or from a replay
            buffer (see `acme.datasets.reverb.make_dataset` documentation).
          observation_network: an optional online network to process observations
            before the policy and the critic.
          target_observation_network: the target observation network.
          policy_optimizer: the optimizer to be applied to the DPG (policy) loss.
          critic_optimizer: the optimizer to be applied to the distributional
            Bellman loss.
          clipping: whether to clip gradients by global norm.
          counter: counter object used to keep track of steps.
          logger: logger object to be used by learner.
          checkpoint: boolean indicating whether to checkpoint the learner.
        """
        if isinstance(replay_client, reverb.TFClient):
            replay_client = reverb.Client(replay_client._server_address)
        self._replay_client = replay_client
        self.per = per
        self.importance_sampling_exponent = importance_sampling_exponent
        self._th = threshold
        self._obj_func = obj_func
        self._critic_loss_func = GPD_quantile_regression

        # Store online and target networks.
        self._policy_network = policy_network
        self._critic_network = critic_network
        self._target_policy_network = target_policy_network
        self._target_critic_network = target_critic_network
        # GPD networks
        self._GPD_scale_network = GPD_scale_network
        self._GPD_shape_network = GPD_shape_network
        self._target_GPD_scale_network = target_GPD_scale_network
        self._target_GPD_shape_network = target_GPD_shape_network

        #### GPD parameters
        self.GPD_threshold=GPD_threshold
        self.n_GPD_samples=n_GPD_samples
        self.quantile_interval=quantile_interval

        # Make sure observation networks are snt.Module's so they have variables.
        self._observation_network = tf2_utils.to_sonnet_module(
            observation_network)
        self._target_observation_network = tf2_utils.to_sonnet_module(
            target_observation_network)

        # General learner book-keeping and loggers.
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger('learner')

        # Other learner parameters.
        self._discount = discount
        self._clipping = clipping

        # Necessary to track when to update target networks.
        self._num_steps = tf.Variable(0, dtype=tf.int32)
        self._target_update_period = target_update_period

        # Batch dataset and create iterator.
        self._base_iterator = dataset_iterator
        self._demo_iterator = demo_dataset_iterator
        self._run_demo = demo_dataset_iterator is not None
        self._demo_step = demo_step
        self._iterator = self._demo_iterator or self._base_iterator

        # Create optimizers if they aren't given.
        self._critic_optimizer = critic_optimizer or snt.optimizers.Adam(1e-4)
        self._policy_optimizer = policy_optimizer or snt.optimizers.Adam(1e-4)
        self._GPD_optimizer = GPD_optimizer or snt.optimizers.Adam(1e-4)

        # Expose the variables.
        policy_network_to_expose = snt.Sequential(
            [self._target_observation_network, self._target_policy_network])
        self._variables = {
            'critic': self._target_critic_network.variables,
            'policy': policy_network_to_expose.variables,
            'GPD_scale': self._target_GPD_scale_network.variables,
            'GPD_shape': self._target_GPD_shape_network.variables,
        }

        # Create a checkpointer and snapshotter objects.
        self._checkpointer = None
        self._snapshotter = None

        if checkpoint:
            self._checkpointer = tf2_savers.Checkpointer(
                directory=checkpoint_folder,
                subdirectory='gpd_learner',
                add_uid=False,
                objects_to_save={
                    'counter': self._counter,
                    'policy': self._policy_network,
                    'critic': self._critic_network,
                    'observation': self._observation_network,
                    'target_policy': self._target_policy_network,
                    'target_critic': self._target_critic_network,
                    'target_observation': self._target_observation_network,
                    'GPD_scale': self._GPD_scale_network,
                    'GPD_shape': self._GPD_shape_network,
                    'policy_optimizer': self._policy_optimizer,
                    'critic_optimizer': self._critic_optimizer,
                    'GPD_optimizer': self._GPD_optimizer,
                    'num_steps': self._num_steps,
                })
            critic_mean = snt.Sequential(
                [self._critic_network, acme_nets.StochasticMeanHead()])
            self._snapshotter = tf2_savers.Snapshotter(
                objects_to_save={
                    'policy': self._policy_network,
                    'critic': critic_mean,
                })

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.
        self._timestamp = None

    @tf.function
    def _step(self) -> Dict[str, tf.Tensor]:
        # Update target network
        online_variables = (
            *self._observation_network.variables,
            *self._critic_network.variables,
            *self._policy_network.variables,
            *self._GPD_scale_network.variables,
            *self._GPD_shape_network.variables,
        )
        target_variables = (
            *self._target_observation_network.variables,
            *self._target_critic_network.variables,
            *self._target_policy_network.variables,
            *self._target_GPD_scale_network.variables,
            *self._target_GPD_shape_network.variables,
        )

        # Make online -> target network update ops.
        if tf.math.mod(self._num_steps, self._target_update_period) == 0:
            for src, dest in zip(online_variables, target_variables):
                dest.assign(src)


        self._num_steps.assign_add(1)

        # Get data from replay (dropping extras if any). Note there is no
        # extra data here because we do not insert any into Reverb.
        sample = next(self._iterator)
        transitions: types.Transition = sample.data  # Assuming ReverbSample.
        keys, probs = sample.info[:2]

        # Cast the additional discount to match the environment discount dtype.
        discount = tf.cast(self._discount, dtype=transitions.discount.dtype)

        with tf.GradientTape(persistent=True) as tape:
            # Maybe transform the observation before feeding into policy and critic.
            # Transforming the observations this way at the start of the learning
            # step effectively means that the policy and critic share observation
            # network weights.
            o_tm1 = self._observation_network(transitions.observation)

            o_t = self._target_observation_network(
                transitions.next_observation)
            # This stop_gradient prevents gradients to propagate into the target
            # observation network. In addition, since the online policy network is
            # evaluated at o_t, this also means the policy loss does not influence
            # the observation network training.
            o_t = tree.map_structure(tf.stop_gradient, o_t)

            # Critic learning.
            q_tm1 = self._critic_network(o_tm1, transitions.action)
            q_t = self._target_critic_network(
                o_t, self._target_policy_network(o_t))   ## (batch_number*qunatiles_number)
            
                        
            #### GPD generation
            target_scale=self._target_GPD_scale_network(o_t, self._target_policy_network(o_t))
            target_shape=self._target_GPD_shape_network(o_t, self._target_policy_network(o_t))

            target_scale = tf.squeeze(target_scale, axis=-1)
            target_shape = tf.squeeze(target_shape, axis=-1)
            target_loc=tf.zeros_like(target_shape)
            target_GPD_dis= GPDDistribution(loc=target_loc, scale=target_scale, concentration=target_shape)   ##(batch_number)

            n_tail_samples=self.n_GPD_samples
            total_samples = int(round(n_tail_samples / (1 - self.GPD_threshold)))  ##1250

            n_body_samples= total_samples-n_tail_samples


            ### target sampling
            n_body_quantiles=quantile_numbers(GPD_threshold=self.GPD_threshold, direction=1, quantile_interval=self.quantile_interval)

            quantile_values = q_t.values  # Extracting the quantile values for body

            body_values = quantile_values[:, -n_body_quantiles:]  ##191


            # When calculating the ceiling for repeats, ensure the result is cast to tf.int32
            repeats = np.ceil(n_body_samples / body_values.shape[-1]).astype(int)
            repeated_body_quantiles = tf.tile(body_values, [1, repeats])
            # body_samples = tf.random.shuffle(repeated_body_quantiles)[:n_body_samples]
            body_samples = repeated_body_quantiles[:n_body_samples]

            

            # Sample from the GPD network for the tail
            tail_samples = target_GPD_dis.sample(n_tail_samples)
            tail_samples = tf.transpose(tail_samples)  #(batch*n_tail_smaples)

            # Shift the GPD samples by the first body quantile value
            first_body_value = body_values[:, 0]
            shifted_tail_samples = -tail_samples + tf.expand_dims(first_body_value, -1)

            # Combine body values and tail samples
            body_samples = tf.stop_gradient(body_samples)
            shifted_tail_samples = tf.stop_gradient(shifted_tail_samples)
            combined_samples = tf.stop_gradient(tf.concat([shifted_tail_samples, body_samples], axis=1))  ##(batch_number*total_samples)`

            #### Critic loss.
            critic_loss = self._critic_loss_func(q_tm1, transitions.reward,
                                                 discount * transitions.discount, combined_samples)
            

            if self.per:
                # PER: importance sampling weights.
                priorities = tf.abs(critic_loss)
                importance_weights = 1.0 / probs
                importance_weights **= self.importance_sampling_exponent
                importance_weights /= tf.reduce_max(importance_weights)
                critic_loss *= tf.cast(importance_weights,
                                       critic_loss.dtype)

            critic_loss = tf.reduce_mean(critic_loss, axis=[0])


            ### Finding GPD loss
            q_t_GPD = self._critic_network(
                o_t, self._target_policy_network(o_t))

            # Generate online GPD
            scale = self._GPD_scale_network(o_t, self._target_policy_network(o_t))
            shape = self._GPD_shape_network(o_t, self._target_policy_network(o_t))
            scale = tf.squeeze(scale, axis=-1)
            shape = tf.squeeze(shape, axis=-1)
            loc = tf.stop_gradient(tf.zeros_like(shape)) 

            # Get samples exceeding the GPD threshold for MLE
            quantiles= q_t_GPD.quantiles

            # Determine the number of body and target  samples
            threshold_index=quantile_numbers(GPD_threshold=self.GPD_threshold, direction=0, quantile_interval=self.quantile_interval)

            # Ensure threshold_index is valid
            threshold_index = tf.minimum(threshold_index, tf.shape(quantiles)[0])   ## 3
            quantile_values = q_t_GPD.values

            # Get the threshold quantile value
            threshold_value = quantile_values[:, threshold_index: threshold_index+1]
            
            # Filter for quantiles exceeding the threshold and adjust by the threshold value
            excess_samples = tf.stop_gradient( -quantile_values[:, : threshold_index] + threshold_value)

            gpd_loss,  has_valid_log_probs = compute_gpd_loss(excess_samples, loc, scale, shape)

            
            # Policy loss
            if self._run_demo:
                # Policy loss MSE supervised learning
                dpg_a_t = self._policy_network(o_t)
                policy_loss = tf.reduce_mean(tf.square(dpg_a_t - transitions.action),
                                             axis=[0])
            else:
                # Actor learning.
                dpg_a_t = self._policy_network(o_t)
                q_actor = self._critic_network(o_t, dpg_a_t)

                actor_quantile_values = q_actor.values  # Extracting the quantile values for body

                actor_body_values = actor_quantile_values[:, -n_body_quantiles:]  ##191
                actor_first_body_value = actor_body_values[:, 0]
                actor_scale = self._GPD_scale_network(o_t, dpg_a_t)
                actor_shape = self._GPD_shape_network(o_t, dpg_a_t)
                actor_scale = tf.squeeze(actor_scale, axis=-1)
                actor_shape = tf.squeeze(actor_shape, axis=-1)
                actor_loc = tf.stop_gradient(tf.zeros_like(actor_shape)) 
                
           
                ##################################################
                VaR_th = tf.expand_dims (q_actor.var(self._th), -1)
                if self._obj_func == 'cvar':

                    ###### Samples for CVaR
                    actor_samples= GPD_samples(loc=actor_loc, scale= actor_scale, shape= actor_shape, 
                                            threshold_values= -VaR_th + tf.expand_dims(actor_first_body_value, -1),
                                            n_samples= n_tail_samples)

                    dpg_q_t  = -tf.expand_dims(actor_samples, -1) + tf.expand_dims(actor_first_body_value, -1)


                # 
                elif self._obj_func == 'var':
                    dpg_q_t=VaR_th
            

                # Actor loss. If clipping is true use dqda clipping and clip the norm.
                dqda_clipping = 1.0 if self._clipping else None
                policy_loss = losses.dpg(
                    dpg_q_t,
                    dpg_a_t,
                    tape=tape,
                    dqda_clipping=dqda_clipping,
                    clip_norm=self._clipping)
                policy_loss = tf.reduce_mean(policy_loss, axis=[0])

        # Get trainable variables.
        policy_variables = self._policy_network.trainable_variables
        gpd_variables= (self._GPD_scale_network.trainable_variables 
                        + self._GPD_shape_network.trainable_variables)
        # Use a dictionary to ensure uniqueness based on variable names
        unique_variables_dict = {var.name: var for var in gpd_variables}
        # Extract the variables from the dictionary to get a list of unique variables
        gpd_variables = list(unique_variables_dict.values())
                         
        critic_variables = (
            # In this agent, the critic loss trains the observation network.
                self._observation_network.trainable_variables +
                self._critic_network.trainable_variables)

        policy_gradients = tape.gradient(policy_loss, policy_variables)
        # Compute gradients for GPD parameters

        def true_fn():
            return tape.gradient(gpd_loss, gpd_variables)

        def false_fn():
            return [tf.zeros_like(variable) for variable in gpd_variables]

        gpd_gradients = tf.cond(has_valid_log_probs, true_fn, false_fn)

        # Compute gradients for critic and apply.
        critic_gradients = tape.gradient(critic_loss, critic_variables)

        # Delete the tape manually because of the persistent=True flag.
        del tape

        # Maybe clip gradients.
        if self._clipping:
            policy_gradients = tf.clip_by_global_norm(policy_gradients, 40.)[0]
            gpd_gradients = tf.clip_by_global_norm(gpd_gradients, 40)[0]
            critic_gradients = tf.clip_by_global_norm(critic_gradients, 40.)[0]


        # Apply gradients.
        self._policy_optimizer.apply(policy_gradients, policy_variables)
        self._GPD_optimizer.apply(gpd_gradients, gpd_variables)
        self._critic_optimizer.apply(critic_gradients, critic_variables)

        # Losses to track.
        return {
            'critic_loss': critic_loss,
            'policy_loss': policy_loss,
            'gpd_loss': gpd_loss,
            'keys': keys,
            'priorities': priorities if self.per else None,
        }
    
    def step(self):
        # Run the learning step.
        if self._num_steps > self._demo_step:
            self._iterator = self._base_iterator
            self._run_demo = False

        fetches = self._step()

        if self.per:
            # Update priorities in replay.
            keys = fetches.pop('keys')
            priorities = fetches.pop('priorities')
            if self._replay_client:
                self._replay_client.mutate_priorities(
                    table=reverb_adders.DEFAULT_PRIORITY_TABLE,
                    updates=dict(zip(keys.numpy(), priorities.numpy())))
        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp = timestamp

        # Update our counts and record it.
        counts = self._counter.increment(steps=1, walltime=elapsed_time)
        fetches.update(counts)

        # Checkpoint and attempt to write the logs.
        if self._checkpointer is not None:
            self._checkpointer.save()
        if self._snapshotter is not None:
            self._snapshotter.save()
        self._logger.write(fetches)

    def get_variables(self, names: List[str]) -> List[List[np.ndarray]]:
        return [tf2_utils.to_numpy(self._variables[name]) for name in names]
