export CUDA_VISIBLE_DEVICES=0

for vol in {1..8}; do
    python run_d3pg.py @run_configs/agents/d3pg.cfg --env_config=run_configs/envs/bsm_vanilla_eval.yaml --logger_prefix=evt/bsm_vanilla/bsm_vol0.$vol/rl --bsm_portfolio.client_trade_poisson_rate=1.0 --bsm_sde.vol=0.$vol --eval_env.vega_ratio=False --train_env.vega_ratio=False
done

for vol in {1..8}; do
    python run_benchmark.py @run_configs/agents/gamma.cfg --env_config=run_configs/envs/bsm_vanilla_eval.yaml --logger_prefix=evt/bsm_vanilla/bsm_vol0.$vol/gamma --bsm_portfolio.client_trade_poisson_rate=1.0 --bsm_sde.vol=0.$vol --eval_env.vega_ratio=False --train_env.vega_ratio=False
done
