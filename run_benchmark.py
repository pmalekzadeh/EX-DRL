import os
import shutil
from pathlib import Path
import yaml

import acme
from acme import wrappers
import acme.utils.loggers as log_utils
import dm_env


from domain.sde import *
from domain.asset.base import *
from domain.asset.portfolio import Portfolio
from env.trade_env import DREnv
from config.config_loader import ConfigLoader, save_args_to_file
from agent.benchmark_agent import DeltaHedgeAgent, GammaHedgeAgent, VegaHedgeAgent
from analysis.gen_stats import generate_stat
import argparse

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--benchmark_name', type=str, default='DeltaHedging', help='Benchmark name - DeltaHedging or GammaHedging (Default DeltaHedging)')
parser.add_argument('--eval_sim', type=int, default=10000, help='Number of evaluation episodes (Default 10000)')
parser.add_argument('--env_config', type=str, default='', help='Environment config (Default None)')
parser.add_argument('--evaluator_seed', type=int, default=4321, help='Evaluation Seed (Default 4321)')
parser.add_argument('--logger_prefix', type=str, default='logs/', help='Prefix folder for logger (Default None)')


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
    environment.logger = make_logger(logger_prefix, label)
    environment = wrappers.GymWrapper(environment)
    # Clip the action returned by the agent to the environment spec.
    environment = wrappers.SinglePrecisionWrapper(environment)
    return environment


def main(argv):
    args, env_cmd_args = parser.parse_known_args(argv)
    work_folder = args.logger_prefix
    if os.path.exists(work_folder):
        shutil.rmtree(work_folder)
    os.makedirs(work_folder, exist_ok=True)
    save_args_to_file(args, os.path.join(work_folder, 'agent.cfg'))
    # Create an environment, grab the spec, and use it to create networks.
    loggers = make_loggers(work_folder=work_folder)
    
    eval_env = make_environment('eval_env', args.env_config, env_cmd_args, args.logger_prefix, seed=args.evaluator_seed)
    # Construct the agent.
    if args.benchmark_name == 'DeltaHedging':
        agent = DeltaHedgeAgent(eval_env) 
    elif args.benchmark_name == 'GammaHedging':
        agent = GammaHedgeAgent(eval_env)
    elif args.benchmark_name == 'VegaHedging':
        agent = VegaHedgeAgent(eval_env)
    else:
        raise NotImplementedError(f'Benchmark {args.benchmark_name} not implemented.')

    eval_loop = acme.EnvironmentLoop(eval_env, agent, label='eval_loop', logger=loggers['eval_loop'])
    eval_loop.run(num_episodes=min(args.eval_sim, len(eval_env.portfolio.sde)))
    print(generate_stat(f'{work_folder}/logs/eval_env/logs.csv',
                        [0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5]))

    Path(f'{work_folder}/ok').touch()


if __name__ == '__main__':
    import sys

    main(sys.argv[1:])

