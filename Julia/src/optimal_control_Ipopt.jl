"""
    solve_PG_OCP_Ipopt(PG_samples::Vector{PG_sample}, phi::Function, g::Function, R, H, J, h_scenario, h_u; J_u=false, x_vec_0=nothing, v_vec=nothing, e_vec=nothing, u_init=nothing, K_pre_solve=0, solver_opts=nothing, print_progress=true)

Solve the optimal control problem of the following form using Ipopt:

``\\min_{u_{0:H},\\; \\overline{J_H}} \\overline{J_H}``

subject to: 
```math
\\begin{aligned}
\\forall k, &\\forall t \\\\
x_{t+1}^{[k]} &= f_{\\theta^{[k]}}(x_t^{[k]}, u_t) + v_t^{[k]}, \\\\
y_{t}^{[k]} &= g_{\\theta^{[k]}}(x_t^{[k]}, u_t) + w_t^{[k]}, \\\\
J_H^{[k]} &= J_H(u_{0:H}, x_{0:H}^{[k]}, y_{0:H}^{[k]}) \\leq \\overline{J_H}, \\\\
h(&u_{0:H},x_{0:H}^{[k]},y_{0:H}^{[k]}) \\leq 0.
\\end{aligned}
```

# Arguments
- `PG_samples`: PG samples
- `phi`: basis functions
- `g`: observation function
- `R`: variance of zero-mean Gaussian measurement noise - only used if e_vec is not passed
- `H`: horizon of the OCP
- `J`: function with input arguments (``u_{1:H}``, ``x_{1:H}``, ``y_{1:H}``) (or ``u_{1:H}`` if `J_u` is set true) that returns the cost to be minimized
- `h_scenario`: function with input arguments (``u_{1:H}``, ``x_{1:H}``, ``y_{1:H}``) that returns the constraint vector belonging to a scenario; a feasible solution must satisfy ``h_{\\mathrm{scenario}} \\leq 0`` for all scenarios.
- `h_u`: function with input argument ``u_{1:H}`` that returns the constraint vector for the control inputs; a feasible solution satisfy ``h_u \\leq 0``.
- `J_u`: set to true if cost depends only on inputs ``u_{1:H}` - this accelerates the optimization
- `x_vec_0`: vector with K * n_x elements containing the initial state of all models - if not provided, the initial states are sampled based on the PGS samples
- `v_vec`: array of dimension n_x x H x K that contains the process noise for all models and all timesteps - if not provided, the noise is sampled based on the PGS samples
- `e_vec`: array of dimension n_y x H x K that contains the measurement noise for all models and all timesteps - if not provided, the noise is sampled based on the provided `R`
- `u_init`: initial guess for the optimal trajectory
- `K_pre_solve`: if `K_pre_solve > 0`, an initial guess for the optimal trajectory is obtained by solving the OCP with only `K_pre_solve < K` models
- `opts`: SolverOptions struct containing options of the solver
- `print_progress`: if set to true, the progress is printed
"""
function solve_PG_OCP_Ipopt(PG_samples::Vector{PG_sample}, phi::Function, g::Function, R, H, J, h_scenario, h_u; J_u=false, x_vec_0=nothing, v_vec=nothing, e_vec=nothing, u_init=nothing, K_pre_solve=0, solver_opts=nothing, print_progress=true)
    # Time optimization.
    optimization_timer = time()

    # Get number of states, etc.
    K = size(PG_samples, 1)
    n_u = size(PG_samples[1].u_m1, 1)
    n_x = size(PG_samples[1].A, 1)
    n_y = size(R, 1)

    # Sample initial state x_vec_0 if not provided.
    if x_vec_0 === nothing
        x_vec_0 = Array{Float64}(undef, n_x * K)
        for k in 1:K
            # Get model.
            A = PG_samples[k].A
            Q = PG_samples[k].Q
            f(x, u) = A * phi(x, u)
            mvn_v_model = MvNormal(zeros(n_x), Q)

            # Sample state at t=-1.
            star = systematic_resampling(PG_samples[k].w_m1, 1)
            x_m1 = PG_samples[k].x_m1[:, star]

            # Propagate.
            x_vec_0[(k-1)*n_x+1:n_x*k] = f(x_m1, PG_samples[k].u_m1) + rand(mvn_v_model)
        end
    end

    # Sample process noise array v_vec if not provided.
    if v_vec === nothing
        v_vec = Array{Float64}(undef, n_x, H, K)
        for k in 1:K
            Q = PG_samples[k].Q
            mvn_v_model = MvNormal(zeros(n_x), Q)
            v_vec[:, :, k] = rand(mvn_v_model, H)
        end
    end

    # Sample measurement noise array e_vec if not provided.
    if e_vec === nothing
        e_vec = Array{Float64}(undef, n_y, H, K)
        mvn_e = MvNormal(zeros(n_y), R)
        for k in 1:K
            e_vec[:, :, k] = rand(mvn_e, H)
        end
    end

    # Determine initialization.
    # If K_pre_solve is provided, a problem that only considers K_pre_solve randomly selected scenarios is solved first to obtain an initialization for the problem with all K scenarios.
    # With a good initialization the runtime of the optimization is reduced.
    if u_init === nothing
        if (0 < K_pre_solve) && (K_pre_solve < K)
            # Sample the K_pre_solve scenarios that are considered for the initialization.
            k_pre_solve = sample(1:K, K_pre_solve)

            # Get the initial states of the considered scenarios.
            x_vec_0_pre_solve = Array{Float64}(undef, n_x * K_pre_solve)
            for k in 1:K_pre_solve
                x_vec_0_pre_solve[(k-1)*n_x+1:n_x*k] = x_vec_0[(k_pre_solve[k]-1)*n_x+1:n_x*k_pre_solve[k]]
            end

            if print_progress
                println("###### Startet pre-solving step")
            end

            # Solve problem that only contains K_pre_solve randomly selected scenarios.
            u_init = solve_PG_OCP_Ipopt(PG_samples[k_pre_solve], phi, g, R, H, J, h_scenario, h_u; J_u=J_u, x_vec_0=x_vec_0_pre_solve, v_vec=v_vec[:, :, k_pre_solve], e_vec=e_vec[:, :, k_pre_solve], K_pre_solve=0, solver_opts=solver_opts, print_progress=print_progress)[1]

            if print_progress
                println("###### Pre-solving step complete, switching back to the original problem")
            end
        else
            u_init = zeros(n_u, H)
        end
    end

    # Determine initial guess for X and Y.
    X_init = Array{Float64}(undef, n_x * K, H + 1) # initial guess for X
    X_init[:, 1] .= x_vec_0
    Y_init = Array{Float64}(undef, n_y * K, H) # initial guess for Y
    for k in 1:K
        A = PG_samples[k].A
        f(x, u) = A * phi(x, u)
        for t in 1:H
            X_init[n_x*(k-1)+1:n_x*k, t+1] = f(X_init[n_x*(k-1)+1:n_x*k, t], u_init[:, t]) + v_vec[:, t, k]
            Y_init[n_y*(k-1)+1:n_y*k, t] = g(X_init[n_x*(k-1)+1:n_x*k, t], u_init[:, t]) + e_vec[:, t, k]
        end
    end

    # Set up OCP.
    OCP = Model(Ipopt.Optimizer)
    @variable(OCP, U[i=1:n_u, j=1:H], start = u_init[i, j])
    @variable(OCP, X[i=1:n_x*K, j=1:H+1], start = X_init[i, j])
    @variable(OCP, Y[i=1:n_y*K, j=1:H], start = Y_init[i, j])

    # Set the initial state.
    for i in 1:n_x*K
        fix(X[i, 1], x_vec_0[i]; force=true)
    end

    # Set options.
    for opt in solver_opts
        set_attributes(OCP, opt)
    end

    if !print_progress
        set_silent(OCP)
    end

    # Define objective
    if J_u # cost function depends only on u, i.e., J_max = J
        @objective(OCP, Min, J(U))
    else
        J_max = @variable(OCP, J_max)
        @objective(OCP, Min, J_max)
        for k in 1:K
            @constraint(OCP, J(U, X[n_x*(k-1)+1:n_x*k, :], Y[n_y*(k-1)+1:n_y*k, :]) <= J_max)
        end
    end

    # Add dynamic and additional constraints for all scenarios.
    for k in 1:K
        # Get current model.
        A = PG_samples[k].A
        f(x, u) = A * phi(x, u)

        # Add dynamic constraints
        for t in 1:H
            @constraint(OCP, X[n_x*(k-1)+1:n_x*k, t+1] .== f(X[n_x*(k-1)+1:n_x*k, t], U[:, t]) + v_vec[:, t, k])
            @constraint(OCP, Y[n_y*(k-1)+1:n_y*k, t] .== g(X[n_x*(k-1)+1:n_x*k, t], U[:, t]) + e_vec[:, t, k])
        end

        # Add scenario constraints.
        @constraint(OCP, h_scenario(U, X[n_x*(k-1)+1:n_x*k, :], Y[n_y*(k-1)+1:n_y*k, :]) .<= 0)
    end

    # Add constraints for the input.
    @constraint(OCP, h_u(U) .<= 0)

    # Add callback that stores barrier parameter mu.
    barrier_param_mu = Float64[]

    function get_mu(
        alg_mod::Cint,
        iter_count::Cint,
        obj_value::Float64,
        inf_pr::Float64,
        inf_du::Float64,
        mu::Float64,
        d_norm::Float64,
        regularization_size::Float64,
        alpha_du::Float64,
        alpha_pr::Float64,
        ls_trials::Cint)
        push!(barrier_param_mu, mu)
        return true
    end

    MOI.set(OCP, Ipopt.CallbackFunction(), get_mu)

    # Solve OCP.
    if print_progress
        println("### Started optimization algorithm")
    end

    optimize!(OCP)
    time_optimization = time() - optimization_timer

    if print_progress
        @printf("### Optimization complete\nRuntime: %.2f s\n", time_optimization)
    end

    u_opt = value.(U)
    x_opt = reshape(value.(X)', (n_x, H + 1, K))
    y_opt = reshape(value.(Y)', (n_y, H, K))
    J_opt = objective_value(OCP)
    solve_successful = is_solved_and_feasible(OCP)
    iterations = MOI.get(OCP, MOI.BarrierIterations())
    mu = barrier_param_mu[end]

    return u_opt, x_opt, y_opt, J_opt, solve_successful, iterations, mu
end

"""
    solve_PG_OCP_Ipopt_greedy_guarantees(PG_samples::Vector{PG_sample}, phi::Function, g::Function, R, H, J, h_scenario, h_u, β; J_u=false, x_vec_0=nothing, v_vec=nothing, e_vec=nothing, u_init=nothing, K_pre_solve=0, solver_opts=nothing, print_progress=true)

Solve the sample-based optimal control problem using Ipopt and determine a support sub-sample with cardinality s via a greedy constraint removal.
Based on the cardinality s, a bound on the probability that the incurred cost exceeds the worst-case cost or that the constraints are violated when the input trajectory u_{0:H} is applied to the unknown system is calculated.

``\\min_{u_{0:H},\\; \\overline{J_H}} \\overline{J_H}``

subject to: 
```math
\\begin{aligned}
\\forall k, &\\forall t \\\\
x_{t+1}^{[k]} &= f_{\\theta^{[k]}}(x_t^{[k]}, u_t) + v_t^{[k]}, \\\\
y_{t}^{[k]} &= g_{\\theta^{[k]}}(x_t^{[k]}, u_t) + w_t^{[k]}, \\\\
J_H^{[k]} &= J_H(u_{0:H}, x_{0:H}^{[k]}, y_{0:H}^{[k]}) \\leq \\overline{J_H}, \\\\
h(&u_{0:H},x_{0:H}^{[k]},y_{0:H}^{[k]}) \\leq 0.
\\end{aligned}
```

# Arguments
- `PG_samples`: PG samples
- `phi`: basis functions
- `g`: observation function
- `R`: variance of zero-mean Gaussian measurement noise - only used if e_vec is not passed
- `H`: horizon of the OCP
- `J`: function with input arguments (``u_{1:H}``, ``x_{1:H}``, ``y_{1:H}``) (or ``u_{1:H}`` if `J_u` is set true) that returns the cost to be minimized
- `h_scenario`: function with input arguments (``u_{1:H}``, ``x_{1:H}``, ``y_{1:H}``) that returns the constraint vector belonging to a scenario; a feasible solution must satisfy ``h_{\\mathrm{scenario}} \\leq 0`` for all scenarios.
- `h_u`: function with input argument ``u_{1:H}`` that returns the constraint vector for the control inputs; a feasible solution satisfy ``h_u \\leq 0``.
- `β`: confidence parameter
- `J_u`: set to true if cost depends only on inputs ``u_{1:H}` - this accelerates the optimization
- `x_vec_0`: vector with K * n_x elements containing the initial state of all models - if not provided, the initial states are sampled based on the PGS samples
- `v_vec`: array of dimension n_x x H x K that contains the process noise for all models and all timesteps - if not provided, the noise is sampled based on the PGS samples
- `e_vec`: array of dimension n_y x H x K that contains the measurement noise for all models and all timesteps - if not provided, the noise is sampled based on the provided `R`
- `u_init`: initial guess for the optimal trajectory
- `K_pre_solve`: if `K_pre_solve > 0`, an initial guess for the optimal trajectory is obtained by solving the OCP with only `K_pre_solve < K` models
- `opts`: SolverOptions struct containing options of the solver
- `print_progress`: if set to true, the progress is printed
"""
function solve_PG_OCP_Ipopt_greedy_guarantees(PG_samples::Vector{PG_sample}, phi::Function, g::Function, R, H, J, h_scenario, h_u, β; J_u=false, x_vec_0=nothing, v_vec=nothing, e_vec=nothing, u_init=nothing, K_pre_solve=0, solver_opts=nothing, print_progress=true)
    # Time first optimization.
    first_solve_timer = time()

    # Get number of states, etc.
    K = size(PG_samples, 1)
    n_u = size(PG_samples[1].u_m1, 1)
    n_x = size(PG_samples[1].A, 1)
    n_y = size(R, 1)

    # Sample initial state x_vec_0 if not provided.
    if x_vec_0 === nothing
        x_vec_0 = Array{Float64}(undef, n_x * K)
        for k in 1:K
            # Get model.
            A = PG_samples[k].A
            Q = PG_samples[k].Q
            f(x, u) = A * phi(x, u)
            mvn_v_model = MvNormal(zeros(n_x), Q)

            # Sample state at t=-1.
            star = systematic_resampling(PG_samples[k].w_m1, 1)
            x_m1 = PG_samples[k].x_m1[:, star]

            # Propagate.
            x_vec_0[(k-1)*n_x+1:n_x*k] = f(x_m1, PG_samples[k].u_m1) + rand(mvn_v_model)
        end
    end

    # Sample process noise array v_vec if not provided.
    if v_vec === nothing
        v_vec = Array{Float64}(undef, n_x, H, K)
        for k in 1:K
            Q = PG_samples[k].Q
            mvn_v_model = MvNormal(zeros(n_x), Q)
            v_vec[:, :, k] = rand(mvn_v_model, H)
        end
    end

    # Sample measurement noise array e_vec if not provided.
    if e_vec === nothing
        e_vec = Array{Float64}(undef, n_y, H, K)
        mvn_e = MvNormal(zeros(n_y), R)
        for k in 1:K
            e_vec[:, :, k] = rand(mvn_e, H)
        end
    end

    num_failed_optimizations = 0 # number of failed optimizations during the computation of the guarantees

    # Solve the OCP once with all constraints to get an initialization for all following optimizations.
    println("### Startet optimization of fully constrained problem")

    u_init, mu = solve_PG_OCP_Ipopt(PG_samples, phi, g, R, H, J, h_scenario, h_u; J_u=J_u, x_vec_0=x_vec_0, v_vec=v_vec, e_vec=e_vec, u_init=u_init, K_pre_solve=K_pre_solve, solver_opts=solver_opts, print_progress=print_progress)[[1, 7]]

    # The OCP is solved again with initialization to obtain the optimal input u_opt.
    solver_opts["mu_init"] = mu
    println("### Startet optimization of fully constrained problem with initialization")
    u_opt, x_opt, y_opt, J_opt, solve_successful, iter = solve_PG_OCP_Ipopt(PG_samples, phi, g, R, H, J, h_scenario, h_u; J_u=J_u, x_vec_0=x_vec_0, v_vec=v_vec, e_vec=e_vec, u_init=u_init, K_pre_solve=K_pre_solve, solver_opts=solver_opts, print_progress=print_progress)[1:6]

    # Determine guarantees.
    # If a feasible u_opt is found, probabilistic constraint satisfaction guarantees are derived by greedily removing constraints to determine a support sub-sample S.
    if solve_successful
        time_first_solve = time() - first_solve_timer

        # Time computation of guarantees.
        guarantees_timer = time()

        # Determine support sub-samples and guarantees for the generalization of the resulting input trajectory.
        println("### Started search for support sub-sample")

        # Reduce number of iterations - if the number of iterations of the original OCP is exceeded, the solution will likely be different, and the optimization can be stopped.
        solver_opts["max_iter"] = 2 * iter

        # Sort scenarios according to the distance to the constraint boundary - removing the scenarios with the largest distance to the constraint boundary first usually yields better results.
        h_scenario_max = Array{Float64}(undef, K) # minimum distance of the scenarios to the constraint boundary
        for i in 1:K
            h_scenario_max[i] = maximum(h_scenario(u_opt, x_opt[:, :, i], y_opt[:, :, i]))
        end
        scenarios_sorted = sortperm(h_scenario_max; rev=true) # sort scenarios

        # Pre-allocate.
        u_opt_temp = Array{Float64}(undef, n_u, H - 1)
        active_scenarios = scenarios_sorted

        # Greedily remove constraints and check whether the solution changes to determine a support sub-sample.
        for i in 1:K
            # Print progress.
            @printf("Startet optimization with new constraint set\nIteration: %i/%i\n", i, K)

            # Temporaily remove the constraints corresponding to the PG samples with index i from the constraint set.
            temp_scenarios = active_scenarios[active_scenarios.!==scenarios_sorted[i]]

            # Get the initial states of the active scenarios.
            x_vec_0_temp = Array{Float64}(undef, n_x * length(temp_scenarios))
            for k in eachindex(temp_scenarios)
                x_vec_0_temp[(k-1)*n_x+1:n_x*k] = x_vec_0[(temp_scenarios[k]-1)*n_x+1:n_x*temp_scenarios[k]]
            end

            # Solve the OCP with reduced constraint set.
            u_opt_temp, J_opt_temp, solve_successful_temp = solve_PG_OCP_Ipopt(PG_samples[temp_scenarios], phi, g, R, H, J, h_scenario, h_u; J_u=J_u, x_vec_0=x_vec_0_temp, v_vec=v_vec[:, :, temp_scenarios], e_vec=e_vec[:, :, temp_scenarios], u_init=u_init, K_pre_solve=0, solver_opts=solver_opts, print_progress=print_progress)[[1, 4, 5]]

            # If the optimization is successful and the solution does not change, permanently remove the constraints corresponding to the PG samples with index i from the constraint set.
            # A valid subsample has the same local minimum. However, since the numerical solver does not reach this minimum exactly, a threshold value is used here to check whether the solutions are the same.
            if solve_successful_temp && all(abs.(u_opt_temp - u_opt) .< 1e-6) && all(abs.(J_opt_temp - J_opt) .< 1e-6)
                active_scenarios = temp_scenarios
            elseif !solve_successful_temp
                @warn "Optimization with temporarily removed constraints failed. Proceeding with next candidate for a support sub-sample."
                num_failed_optimizations += 1
            end
        end

        # Determine the cardinality of the support sub-sample.
        s = length(active_scenarios)

        # Based on the cardinality of the support sub-sample, determine the parameter ϵ. 
        # 1-ϵ corresponds to a bound on the probability that the incurred cost exceeds the worst-case cost or that the constraints are violated when the input trajectory u_{0:H} is applied to the unknown system.
        epsilon_prob = epsilon(s, K, β)
        epsilon_perc = epsilon_prob * 100

        # Print s, ϵ, and runtime.
        time_guarantees = time() - guarantees_timer
        @printf("### Support sub sample found\nCardinality of the support sub-sample (s): %i\nMax. constraint violation probability (1-epsilon): %.2f %%\nTime to compute u*: %.2f s\nTime to compute 1-epsilon: %.2f s\n", s, 100 - epsilon_perc, time_first_solve, time_guarantees)
    else
        # In case the initial problem is infeasible, skip the computation of guarantees.
        @warn "No feasible solution found for the initial problem. Skipping computation of guarantees."
        time_first_solve = NaN
        J_opt = NaN
        s = NaN
        epsilon_prob = NaN
        epsilon_perc = NaN
        time_guarantees = NaN
    end
    return u_opt, x_opt, y_opt, J_opt, s, epsilon_prob, epsilon_perc, time_first_solve, time_guarantees, num_failed_optimizations
end