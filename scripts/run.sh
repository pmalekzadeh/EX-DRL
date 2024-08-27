python run_d3pg.py @run_configs/agents/d3pg.cfg --env_config=run_configs/test_envs/train_bsm_0.3_test_bsm.yaml --logger_prefix=logs/bsm/train_bsm_0.3_test_bsm
python run_benchmark.py @run_configs/agents/gamma.cfg --env_config=run_configs/test_envs/train_bsm_0.3_test_bsm.yaml --logger_prefix=logs/bsm/gamma

python run_d3pg.py @run_configs/agents/d3pg.cfg --env_config=run_configs/test_envs/train_bsm_0.3_test_sabr.yaml --logger_prefix=logs/sabr/train_bsm_0.3_test_realdata --eval_only --agent_path=logs/bsm/train_bsm_0.3_test_bsm
python run_benchmark.py @run_configs/agents/gamma.cfg --env_config=run_configs/test_envs/train_bsm_0.3_test_sabr.yaml --logger_prefix=logs/sabr/gamma

python run_d3pg.py @run_configs/agents/d3pg.cfg --env_config=run_configs/test_envs/train_bsm_0.3_test_realdata.yaml --logger_prefix=logs/realdata/train_bsm_0.3_test_realdata --eval_only --agent_path=logs/bsm/train_bsm_0.3_test_bsm
python run_benchmark.py @run_configs/agents/gamma.cfg --env_config=run_configs/test_envs/train_bsm_0.3_test_realdata.yaml --logger_prefix=logs/realdata/gamma