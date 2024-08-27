import yaml
import ast
import shlex
from copy import deepcopy

"""Load config from yaml file and create objects from config

Example:
# BSM Environment
# Single Client Call option with ttm = 40 days
# Hedging options are daily ATM call with ttm = 20 days
# Lasting until client call option expires with episode_length = 40 days
# SDE is BSM with volatility = 0.3 and s = 10.0

test_env:
  class_name: DREnv
  params:
    portfolio:
      ref: bsm_portfolio
    episode_length: 40
    vega_state: True
    scale_action: True
    action_low: 0
    action_high: 1
    logger_folder: test_dr/bsm_vanilla_env

bsm_portfolio:
  class_name: Portfolio
  params:
    sde:
      ref: bsm_sde
    client_trade_poisson_rate: 0.0
    client_options:
      class_name: VanillaOption
      params:
        sde:
          ref: bsm_sde
        call: [True]
        moneyness: [1.0]
        ttm: [40]
        shares: [1.0]
    hedging_options:
      class_name: VanillaOption
      params:
        sde:
          ref: bsm_sde
        sim_moneyness_mean: 1.0
        sim_moneyness_std: 0.0
        sim_ttms: [20]
        sim_call: [True]

bsm_sde:
  class_name: BSMSimulator
  params:
    s: 10.0
    vol: 0.3

"""
class ConfigLoader:
    registered_classes = {}

    def __init__(self, config_data=None, config_file='', cmd_args=[]):
        self.config_file = config_file
        self.config_data = self.argparser(config_data, cmd_args) if config_data \
            else self.load_config(cmd_args)
        self.config_data_dict = deepcopy(self.config_data)
        self.objects = {}

    def load_config(self, cmd_args):
        if self.config_file:
            with open(self.config_file, "r") as file:
                config_data = yaml.safe_load(file)
        else:
            config_data = {}
        if cmd_args:
            config_data = self.argparser(config_data, cmd_args)
        return config_data

    def save_config(self, config_file):
        with open(config_file, "w") as file:
            yaml.dump(self.config_data_dict, file)

    def load_objects(self):
        for name, obj_config in self.config_data.items():
            self.create_or_get_object(name, obj_config)

    def register_class(cls):
        ConfigLoader.registered_classes[cls.__name__] = cls
        return cls

    def argparser(self, config_data, cmd_args):
        args = cmd_args
        for arg in args:
            key, value = arg.split('=')
            key = key.lstrip('-')
            keys = key.split('.')
            last_key = keys[-1]
            temp = config_data
            for key in keys:
                while key not in temp:
                    if 'params' in temp:
                        temp = temp['params']
                    if 'ref' in temp:
                        temp = config_data[temp['ref']]
                    assert isinstance(temp, dict), f"Invalid config: {temp} while searching for {key}"
                if last_key not in temp:
                    temp = temp[key]
            if isinstance(temp, dict):
                temp[last_key] = ast.literal_eval(value)
        return config_data

    def __getitem__(self, name):
        return self.create_or_get_object(name, self.config_data[name])

    def _is_obj(self, obj_dict):
        if not isinstance(obj_dict, dict):
            return False
        return 'ref' in obj_dict or 'class_name' in obj_dict

    def create_or_get_object(self, name, obj_config):
        if name in self.objects:
            return self.objects[name]
        
        if obj_config.get("ref"):
            # use reference object from global config
            name = obj_config["ref"]
            if name not in self.config_data:
                raise ValueError(f"Reference object {name} not found in config")

            obj_config = self.config_data[obj_config["ref"]]
            return self.create_or_get_object(name, obj_config)

        # create object
        class_name = obj_config["class_name"]
        params = obj_config.get("params", {})

        for param_name, param_value in params.items():
            # if it is a dictionary, recursively create or get the object
            if isinstance(param_value, dict):
                param_value = self.create_or_get_object(name + "." + param_name, param_value)
            if isinstance(param_value, list) and self._is_obj(param_value[0]):
                param_value = [self.create_or_get_object(name + "." + param_name, item) for item in param_value]
            params[param_name] = param_value

        obj = ConfigLoader.registered_classes[class_name](**params)
        self.objects[name] = obj
        return obj

def save_args_to_file(args, filename='args.txt'):
    with open(filename, 'w') as file:
        # Write arguments in a way they can be read later. 
        # For example, as '--arg1 value --arg2 value' etc.
        args_dict = vars(args)
        for arg_key, arg_value in args_dict.items():
            file.write('--{} {}\n'.format(arg_key, shlex.quote(str(arg_value))))
