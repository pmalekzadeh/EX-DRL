for vol in {1..8}; do
    python run_benchmark.py @run_configs/agents/gamma.cfg --env_config=run_configs/envs/bsm_vanilla_eval.yaml --logger_prefix=policies/bsm_vanilla/dr_vol/gamma/bsm_vol0.$vol --bsm_portfolio.client_trade_poisson_rate=1.0 --bsm_sde.vol=0.$vol
    python run_benchmark.py @run_configs/agents/vega.cfg --env_config=run_configs/envs/bsm_vanilla_eval.yaml --logger_prefix=policies/bsm_vanilla/dr_vol/vega/bsm_vol0.$vol --bsm_portfolio.client_trade_poisson_rate=1.0 --bsm_sde.vol=0.$vol
done
