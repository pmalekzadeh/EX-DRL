export CUDA_VISIBLE_DEVICES=1  


 for vol in {1..8}; do
     python run_d3pg.py @run_configs/agents/d3pg.cfg --env_config=run_configs/envs/bsm_vanilla_eval.yaml --logger_prefix=logs/bsm_vanilla/EX-DRL/bsm_vol0.$vol/rl --bsm_portfolio.client_trade_poisson_rate=1.0 --bsm_sde.vol=0.$vol --eval_env.vega_ratio=False --train_env.vega_ratio=False
 done






