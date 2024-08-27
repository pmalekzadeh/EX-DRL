for vol in {1..8}; do
    # python run_benchmark.py @run_configs/agents/gamma.cfg --env_config=run_configs/envs/dr_mixsde_vanilla.yaml --logger_prefix=policies/bsm_vanilla/dr_mixsde/gamma/sabr_volvol0.$vol --sabr_portfolio.client_trade_poisson_rate=1.0 --sabr_sde.volvol=0.$vol
    python run_benchmark.py @run_configs/agents/vega.cfg --env_config=run_configs/envs/dr_mixsde_vanilla.yaml --logger_prefix=policies/bsm_vanilla/dr_mixsde/vega/sabr_volvol0.$vol --sabr_portfolio.client_trade_poisson_rate=1.0 --sabr_sde.volvol=0.$vol
done
