# List of fuctions

```@docs
particle_Gibbs(u_training, y_training, K, K_b, k_d, N, phi::Function, Lambda_Q, ell_Q, Q_init, V, A_init, x_init_dist, g, R; x_prim=nothing)
solve_PG_OCP_Altro(PG_samples::Vector{PG_sample}, phi::Function, R, H, u_min, u_max, y_min, y_max, R_cost_diag; x_vec_0=nothing, v_vec=nothing, e_vec=nothing, u_init=nothing, K_pre_solve=0, active_constraints=nothing, opts=nothing, print_progress=true)
solve_PG_OCP_Altro_greedy_guarantees(PG_samples::Vector{PG_sample}, phi::Function, R, H, u_min, u_max, y_min, y_max, R_cost_diag, β; x_vec_0=nothing, v_vec=nothing, e_vec=nothing, u_init=nothing, K_pre_solve=0, opts=nothing, print_progress=true)
solve_PG_OCP_Ipopt(PG_samples::Vector{PG_sample}, phi::Function, g::Function, R, H, J, h_scenario, h_u; J_u=false, x_vec_0=nothing, v_vec=nothing, e_vec=nothing, u_init=nothing, K_pre_solve=0, solver_opts=nothing, print_progress=true)
solve_PG_OCP_Ipopt_greedy_guarantees(PG_samples::Vector{PG_sample}, phi::Function, g::Function, R, H, J, h_scenario, h_u, β; J_u=false, x_vec_0=nothing, v_vec=nothing, e_vec=nothing, u_init=nothing, K_pre_solve=0, solver_opts=nothing, print_progress=true)
test_prediction(PG_samples::Vector{PG_sample}, phi::Function, g, R, k_n, u_test, y_test)
plot_predictions(y_pred, y_test; plot_percentiles=false, y_min=nothing, y_max=nothing)
plot_autocorrelation(PG_samples::Vector{PG_sample}; max_lag=0)
epsilon(s::Int64, K::Int64, β::Float64)
MNIW_sample(Phi, Psi, Sigma, V, Lambda_Q, ell_Q, T)
systematic_resampling(W, N)
```
