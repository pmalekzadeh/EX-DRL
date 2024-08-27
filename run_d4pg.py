import os
import shutil
from pathlib import Path
import yaml
from typing import Mapping, Sequence

import tensorflow as tf
import acme
from acme import specs
from acme import types
from acme import wrappers
from acme.agents.tf import actors
from acme.tf import networks
from acme.tf import utils as tf2_utils
from acme.tf.savers import make_snapshot
import acme.utils.loggers as log_utils
import dm_env
import numpy as np
import sonnet as snt
import pandas as pd

from domain.sde import *
from domain.asset.base import *
from domain.asset.portfolio import Portfolio
from env.trade_env import DREnv
from config.config_loader import ConfigLoader, save_args_to_file
from agent.agent import  GPD
# Note that the GammaVegaTradingEnv for Exotic Options are never implemented, right now we are only studying the delta hedging behaviours.
import agent.distributional as ad
from analysis.gen_stats import generate_stat
from agent.demonstrations import DemonstrationRecorder

import argparse

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--train_sim', type=int, default=50000, help='Number of training episodes')
parser.add_argument('--eval_sim', type=int, default=10000, help='Number of evaluation episodes')
parser.add_argument('--eval_only', action='store_true', help='Evaluation only')
parser.add_argument('--continue_train', action='store_true', help='Continue training')
parser.add_argument('--agent_path', type=str, default='', help='Agent Path')
parser.add_argument('--env_config', type=str, default='', help='Environment config')
parser.add_argument('--actor_seed', type=int, default=1234, help='Actor seed')
parser.add_argument('--evaluator_seed', type=int, default=4321, help='Evaluation Seed')
parser.add_argument('--n_step', type=int, default=5, help='DRL TD Nstep')
parser.add_argument('--critic', type=str, default='qr', help='critic distribution type - c51 or qr')
parser.add_argument('--obj_func', type=str, default='var', help='Objective function select from meanstd, var or cvar')
parser.add_argument('--std_coef', type=float, default=1.645, help='Std coefficient when obj_func=meanstd.')
parser.add_argument('--threshold', type=float, default=0.99, help='Objective function threshold.')
parser.add_argument('--GPD_threshold', type=float, default=0.96, help='Tail v.s. body threshold.')
parser.add_argument('--n_GPD_samples', type=int, default=48, help='Tail samples.')
parser.add_argument('--heavy_tail', type=bool, default=True, help='Heavy-tailed or light-tailed environment.')
parser.add_argument('--quantile_interval', type=float, default=0.01, help='Quantile interval')
parser.add_argument('--logger_prefix', type=str, default='logs/', help='Prefix folder for logger')
parser.add_argument('--per', action='store_true', help='Use PER for Replay sampling')
parser.add_argument('--importance_sampling_exponent', type=float, default=0.2, help='importance sampling exponent for updating importance weight for PER')
parser.add_argument('--priority_exponent', type=float, default=0.6, help='priority exponent for the Prioritized replay table')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for optimizer')
parser.add_argument('--GPD_lr', type=float, default=1e-6, help='Learning rate for optimizer')
parser.add_argument('--sigma', type=float, default=0.3, help='Sigma Noise for exploration')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size to train the Network')
parser.add_argument('--buffer_steps', type=int, default=0, help='Buffer Steps in Transaction Cost Case')
parser.add_argument('--demo_path', type=str, default='', help='Demo Path')
parser.add_argument('--demo_step', type=int, default=1000, help='Demo Steps')

def make_logger(work_folder, label, terminal=False):
    loggers = [
        log_utils.CSVLogger(work_folder,
                            label=label, add_uid=False)
    ]
    if terminal:
        loggers.append(log_utils.TerminalLogger(label=label, print_fn=print))

    logger = log_utils.Dispatcher(loggers, log_utils.to_numpy)
    logger = log_utils.NoneFilter(logger)
    return logger


def make_loggers(work_folder):
    return dict(
        train_loop=make_logger(work_folder, 'train_loop', terminal=True),
        eval_loop=make_logger(work_folder, 'eval_loop', terminal=True),
        learner=make_logger(work_folder, 'learner')
    )


def make_environment(label, env_config_file, env_cmd_args, logger_prefix, seed=1234) -> dm_env.Environment:
    # Make sure the environment obeys the dm_env.Environment interface.
    config_loader = ConfigLoader(config_file=env_config_file, cmd_args=env_cmd_args)
    config_loader.load_objects()
    config_loader.save_config(os.path.join(logger_prefix, 'env.yaml'))
    environment: DREnv = config_loader[label] if label in config_loader.objects else config_loader['env']
    environment.seed(seed)
    environment.logger = make_logger(logger_prefix, label) if label == 'eval_env' else None
    environment = wrappers.GymWrapper(environment)
    # Clip the action returned by the agent to the environment spec.
    environment = wrappers.CanonicalSpecWrapper(environment, clip=True)
    environment = wrappers.SinglePrecisionWrapper(environment)
    return environment

# The default settings in this network factory will work well for the
# TradingENV task but may need to be tuned for others. In
# particular, the vmin/vmax and num_atoms hyperparameters should be set to
# give the distributional critic a good dynamic range over possible discounted
# returns. Note that this is very different than the scale of immediate rewards.


def make_networks(
    action_spec: specs.BoundedArray,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    vmin: float = -150.,
    vmax: float = 150.,
    num_atoms: int = 51,
) -> Mapping[str, types.TensorTransformation]:
    """Creates the networks used by the agent."""

    # Get total number of action dimensions from action spec.
    num_dimensions = np.prod(action_spec.shape, dtype=int)

    # Create the shared observation network; here simply a state-less operation.
    observation_network = tf2_utils.batch_concat

    # Create the policy network.
    policy_network = snt.Sequential([
        networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
        networks.NearZeroInitializedLinear(num_dimensions),
        networks.TanhToSpec(action_spec),
    ])

    # Create the critic network.
    critic_network = snt.Sequential([
        # The multiplexer concatenates the observations/actions.
        networks.CriticMultiplexer(),
        networks.LayerNormMLP(critic_layer_sizes, activate_final=True),
        ad.RiskDiscreteValuedHead(vmin, vmax, num_atoms),
    ])

    return {
        'policy': policy_network,
        'critic': critic_network,
        'observation': observation_network,
    }


def make_quantile_networks(
    action_spec: specs.BoundedArray,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    quantile_interval: float = 0.01
) -> Mapping[str, types.TensorTransformation]:
    """Creates the networks used by the agent."""

    # Get total number of action dimensions from action spec.
    num_dimensions = np.prod(action_spec.shape, dtype=int)

    # Create the shared observation network; here simply a state-less operation.
    observation_network = tf2_utils.batch_concat

    # Create the policy network.
    policy_network = snt.Sequential([
        networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
        networks.NearZeroInitializedLinear(num_dimensions),
        networks.TanhToSpec(action_spec),
    ])
    quantiles = np.arange(quantile_interval, 1.0, quantile_interval)
    # Create the critic network.
    critic_network = snt.Sequential([
        # The multiplexer concatenates the observations/actions.
        networks.CriticMultiplexer(),
        networks.LayerNormMLP(critic_layer_sizes, activate_final=True),
        ad.QuantileDiscreteValuedHead(
            quantiles=quantiles, prob_type=ad.QuantileDistProbType.MID),
    ])

    return {
        'policy': policy_network,
        'critic': critic_network,
        'observation': observation_network,
    }

################### GPD network
def make_GPD_networks(
    action_spec: specs.BoundedArray,
    policy_layer_sizes: Sequence[int] = (256, 256, 256),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    quantile_interval: float = 0.005,
    init_scale: float = 0.3,
    min_scale: float = 1e-6,
    GPD_layer_sizes: Sequence[int] = (512,  512, 256),
    heavy_tail: bool =True
) -> Mapping[str, types.TensorTransformation]:
    """Creates the networks used by the agent."""

    # Get total number of action dimensions from action spec.
    num_dimensions = np.prod(action_spec.shape, dtype=int)

    # Create the shared observation network; here simply a state-less operation.
    observation_network = tf2_utils.batch_concat

    # Create the policy network.
    policy_network = snt.Sequential([
        networks.LayerNormMLP(policy_layer_sizes, activate_final=True),
        networks.NearZeroInitializedLinear(num_dimensions),
        networks.TanhToSpec(action_spec),
    ])
    quantiles = np.arange(quantile_interval, 1.0, quantile_interval)

    # Create the critic network.
    critic_network = snt.Sequential([
        # The multiplexer concatenates the observations/actions.
        networks.CriticMultiplexer(),
        networks.LayerNormMLP(critic_layer_sizes, activate_final=True),
        ad.QuantileDiscreteValuedHead(
            quantiles=quantiles, prob_type=ad.QuantileDistProbType.MID),
    ])

    # Create the GPD network.
    gpd_head  = ad.GPDDistributionHead(
        backbone_layer_sizes=GPD_layer_sizes,
          init_scale = init_scale, min_scale= min_scale, heavy_tail=heavy_tail)

    return {
        'policy': policy_network,
        'critic': critic_network,
        'observation': observation_network,
        'GPD_shape': gpd_head.shape_network,
        'GPD_scale': gpd_head.scale_network,
    }


def save_policy(policy_network, checkpoint_folder):
    snapshot = make_snapshot(policy_network)
    tf.saved_model.save(snapshot, checkpoint_folder+'/policy')


def load_policy(policy_network, checkpoint_folder):
    trainable_variables_snapshot = {}
    load_net = tf.saved_model.load(checkpoint_folder+'/policy')
    for var in load_net.trainable_variables:
        trainable_variables_snapshot['/'.join(
            var.name.split('/')[1:])] = var.numpy()
    for var in policy_network.trainable_variables:
        var_name_wo_name_scope = '/'.join(var.name.split('/')[1:])
        var.assign(
            trainable_variables_snapshot[var_name_wo_name_scope])


def main(argv):
    args, env_cmd_args = parser.parse_known_args(argv)
    work_folder = args.logger_prefix
    if os.path.exists(work_folder):
        shutil.rmtree(work_folder)
    os.makedirs(work_folder, exist_ok=True)
    save_args_to_file(args, os.path.join(work_folder, 'agent.cfg'))
    
    # Create an environment, grab the spec, and use it to create networks.
    environment = make_environment('train_env', args.env_config, env_cmd_args, args.logger_prefix, args.actor_seed)
    environment_spec = specs.make_environment_spec(environment)
   
    agent_networks = make_GPD_networks( quantile_interval=args.quantile_interval,
            action_spec=environment_spec.actions, heavy_tail=args.heavy_tail)

    loggers = make_loggers(work_folder=work_folder)
    # Construct the agent.
    if args.critic == 'gpd':
        agent = GPD(
            obj_func=args.obj_func,
            threshold=args.threshold,
            GPD_threshold=args.GPD_threshold,   ## GPD_threshold
            n_GPD_samples=args.n_GPD_samples,
            quantile_interval=args.quantile_interval,
            environment_spec=environment_spec,
            policy_network=agent_networks['policy'],
            critic_network=agent_networks['critic'],
            observation_network=agent_networks['observation'],
            scale_network=agent_networks['GPD_scale'],
            shape_network=agent_networks['GPD_shape'],
            n_step=args.n_step,
            discount=1.0,
            sigma=0.3,  # pytype: disable=wrong-arg-types
            checkpoint=False,
            logger=loggers['learner'],
            batch_size=args.batch_size,
            per=args.per,
            priority_exponent=args.priority_exponent,
            importance_sampling_exponent=args.importance_sampling_exponent,
            policy_optimizer=snt.optimizers.Adam(args.lr),
            critic_optimizer=snt.optimizers.Adam(args.lr),
            GPD_optimizer=snt.optimizers.Adam(args.GPD_lr),
            demonstration_dataset=None if args.demo_path == ''
                else DemonstrationRecorder().load(args.demo_path).make_tf_dataset(),
            demonstration_step=args.demo_step,
        )

    # Create the evaluation policy.
    if args.eval_only or args.continue_train:
        policy_net = agent._learner._policy_network
        if args.agent_path == '':
            load_policy(policy_net, work_folder)
        else:
            load_policy(policy_net, args.agent_path)
        eval_policy = snt.Sequential([
            agent_networks['observation'],
            policy_net,
        ])
    else:
        eval_policy = snt.Sequential([
            agent_networks['observation'],
            agent_networks['policy'],
        ])
        
    # Create the environment loop used for training.
    if not args.eval_only:
        train_loop = acme.EnvironmentLoop(
            environment, agent, label='train_loop', logger=loggers['train_loop'])
        print(min(args.train_sim, len(environment.portfolio.sde)))
        train_loop.run(num_episodes=min(args.train_sim, len(environment.portfolio.sde)))
        save_policy(agent._learner._policy_network, work_folder)


    # Create the evaluation actor and loop.
    eval_actor = actors.FeedForwardActor(policy_network=eval_policy)
    eval_env = make_environment('eval_env', args.env_config, env_cmd_args, args.logger_prefix, seed=args.evaluator_seed)
    eval_loop = acme.EnvironmentLoop(eval_env, eval_actor, label='eval_loop', logger=loggers['eval_loop'])
    eval_loop.run(num_episodes=min(args.eval_sim, len(eval_env.portfolio.sde)))
    print(generate_stat(f'{work_folder}/logs/eval_env/logs.csv',
                       [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5]))

    Path(f'{work_folder}/ok').touch()


if __name__ == '__main__':
    # import cProfile
    # cProfile.run('app.run(main)', 'output.prof')
    import sys

    main(sys.argv[1:])
