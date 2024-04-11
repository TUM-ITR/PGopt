include("particle_Gibbs.jl")
using Altro
using TrajectoryOptimization
using RobotDynamics
using ForwardDiff
using FiniteDiff
using SpecialFunctions

# Define RobotDynamics object - see RobotDynamics.jl for explanations.
# The PG samples are combined into a single model with state x_vec containing the K*n_x states of the individual models.
RobotDynamics.@autodiff struct PGS_model_obj <: RobotDynamics.DiscreteDynamics
    PG_samples::Vector{PG_sample} # PG samples
    K::Int64 # number of samples
    phi::Function # basis functions
    n_x::Int64 # number of states of a single PGS sample
    n_u::Int64 # number of control inputs
    v_vec::Array{Float64,3} # array containing process noise - the process noise is sampled only once
end

# Define copy().
Base.copy(PGS_model::PGS_model_obj) = PGS_model_obj(PGS_model.PG_samples, PGS_model.K, PGS_model.phi, PGS_model.n_x, PGS_model.n_u, PGS_model.v_vec)

RobotDynamics.state_dim(PGS_model::PGS_model_obj) = PGS_model.K * PGS_model.n_x # the PG samples are combined into a single model with K*n_x states
RobotDynamics.control_dim(PGS_model::PGS_model_obj) = PGS_model.n_u
RobotDynamics.default_signature(::PGS_model_obj) = RobotDynamics.InPlace() # use in-place computation

"""
    RobotDynamics.discrete_dynamics(PGS_model::PGS_model, x_vec, u, t, dt)

Compute the state of all scenarios in the next time step t+1. The PG samples are combined into a single model with K*n_x states.

# Arguments
- `PGS_model`: RobotDynamics object
- `x_vec`: vector with K*n_x elements containing the state of all models in the current timestep (t)
- `u`: current control input
- `t`: time index
- `dt`: step size - unused
"""
function RobotDynamics.discrete_dynamics(PGS_model::PGS_model_obj, x_vec, u, t, dt)
    # Initialize x_vec_t+1.
    x_vec_n = Array{Any}(undef, size(x_vec))

    # Compute the state x_vec_t+1 for all K models.
    for k in 1:PGS_model.K
        x_vec_n[PGS_model.n_x*(k-1)+1:PGS_model.n_x*k] = PGS_model.PG_samples[k].A * PGS_model.phi(x_vec[PGS_model.n_x*(k-1)+1:PGS_model.n_x*k], u) + PGS_model.v_vec[:, round(Int32, t)+1, k]
    end
    return x_vec_n # return x_t+1
end

"""
    RobotDynamics.discrete_dynamics!(PGS_model::PGS_model, x_vec_n, x_vec, u, t, dt)

Compute the state of all scenarios in the next time step (t+1) in-place. The PG samples are combined into a single model with K*n_x states.

# Arguments
- `x_vec`: vector with K*n_x elements containing the state of all models in the next timestep (t+1)
- `x_vec`: vector with K*n_x elements containing the state of all models in the current timestep (t)
- `u`: current control input
- `t`: time index
- `dt`: step size - unused
"""
function RobotDynamics.discrete_dynamics!(PGS_model::PGS_model_obj, x_vec_n, x_vec, u, t, dt)
    for k in 1:PGS_model.K
        x_vec_n[PGS_model.n_x*(k-1)+1:PGS_model.n_x*k] = PGS_model.PG_samples[k].A * PGS_model.phi(x_vec[PGS_model.n_x*(k-1)+1:PGS_model.n_x*k], u) + PGS_model.v_vec[:, round(Int32, t)+1, k]
    end
    return nothing
end

"""
    solve_PG_OCP(PG_samples::Vector{PG_sample}, phi::Function, R, H, u_min, u_max, y_min, y_max, R_cost_diag; x_vec_0=nothing, v_vec=nothing, e_vec=nothing, u_init=nothing, K_pre_solve=0, active_constraints=nothing, opts=nothing, print_progress=true)

Solve the optimal control problem of the following form:

min ∑_{∀t} 1/2 * u_t * Diagonal(R_cost_diag) * u_t

subject to: ∀k, ∀t
    x_{t+1}^k = f(x_t^k, u_t^k) + v_t^k - implemented in RobotDynamics.discrete_dynamics
    x_t^k[1:n_y] >= y_{min,t} - e_t^k
    x_t^k[1:n_y] <= y_{max,t} - e_t^k
    u_t >= u_{min,t}
    u_t <= u_{max,t}.

Note that the output constraints imply the measurement function y_t^k = x_t^k[1:n_y].
Further note that the states of the individual models x^k are combined in the vector x_vec of dimension K*n_x.

# Arguments
- `PG_samples`: PG samples
- `phi`: basis functions
- `R`: variance of zero-mean Gaussian measurement noise - only used if e_vec is not passed
- `H`: horizon of the OCP
- `u_min`: array of dimension 1 x 1 or 1 x H containing the minimum control input for all timesteps
- `u_max`: array of dimension 1 x 1 or 1 x H containing the maximum control input for all timesteps
- `y_min`: array of dimension 1 x 1 or 1 x H containing the minimum system output for all timesteps
- `y_max`: array of dimension 1 x 1 or 1 x H containing the maximum system output for all timesteps
- `R_cost_diag`: parameter of the diagonal quadratic cost function
- `x_vec_0`: vector with K*n_x elements containing the initial state of all models - if not provided, the initial states are sampled based on the PGS samples
- `v_vec`: array of dimension n_x x H x K that contains the process noise for all models and all timesteps - if not provided, the noise is sampled based on the PGS samples
- `e_vec`: array of dimension n_y x H x K that contains the measurement noise for all models and all timesteps - if not provided, the noise is sampled based on the provided R
- `u_init`: initial guess for the optimal trajectory
- `K_pre_solve`: if K_pre_solve > 0, an initial guess for the optimal trajectory is obtained by solving the OCP with only K_pre_solve < K models
- `active_constraints`: vector containing the indices of the models, for which the output constraints are active - if not provided, the output constraints are considered for all models
- `opts`: SolverOptions struct containing options of the solver
- `print_progress`: if set to true, the progress is printed
"""
function solve_PG_OCP(PG_samples::Vector{PG_sample}, phi::Function, R, H, u_min, u_max, y_min, y_max, R_cost_diag; x_vec_0=nothing, v_vec=nothing, e_vec=nothing, u_init=nothing, K_pre_solve=0, active_constraints=nothing, opts=nothing, print_progress=true)
    # Time optimization.
    optimization_timer = time()

    # Get number of states, etc.
    K = size(PG_samples, 1)
    n_u = size(u_max, 1)
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

    # If active_constraints is not provided, the output constraints are considered for all models.
    if active_constraints === nothing
        active_constraints = Array(1:K)
    end

    # Determine initialization.
    # If K_pre_solve is provided, a problem that only considers K_pre_solve randomly selected scenarios is solved first to obtain an initialization for the problem with all K scenarios.
    # With a good initialization the runtime of the optimization is reduced.
    if u_init === nothing
        if (0 < K_pre_solve) && (K_pre_solve < K)
            # Sample the K_pre_solve scenarios that are considered for the initialization.
            k_pre_solve = sample(active_constraints, K_pre_solve)

            # Get the initial states of the considered scenarios.
            x_vec_0_pre_solve = Array{Float64}(undef, n_x * K_pre_solve)
            for k in 1:K_pre_solve
                x_vec_0_pre_solve[(k-1)*n_x+1:n_x*k] = x_vec_0[(k_pre_solve[k]-1)*n_x+1:n_x*k_pre_solve[k]]
            end

            if print_progress
                println("###### Startet pre-solving step")
            end

            # Solve problem that only contains K_pre_solve randomly selected scenarios.
            u_init, max_penalty = solve_PG_OCP(PG_samples[k_pre_solve], phi, R, H, u_min, u_max, y_min, y_max, R_cost_diag; x_vec_0=x_vec_0_pre_solve, v_vec=v_vec[:, :, k_pre_solve], e_vec=e_vec[:, :, k_pre_solve], K_pre_solve=0, opts=opts, print_progress=print_progress)[[2, 8]]

            # Increase penalty - this can further decrease the runtime but might be unstable.
            # opts.penalty_initial = 10*opts.penalty_initial
            # opts.penalty_initial = max_penalty

            if print_progress
                println("###### Pre-solving step complete, switching back to the original problem")
            end
        end
    end

    # Set solver options if not provided - see Altro.jl documentation.
    if opts === nothing
        opts = SolverOptions()
        opts.constraint_tolerance = 1e-5
        opts.cost_tolerance = 1e-3
        opts.cost_tolerance_intermediate = 10 * opts.cost_tolerance
        opts.projected_newton_tolerance = sqrt(opts.constraint_tolerance)
        if u_init !== nothing
            # Increase penalty in case an initial guess is provided.
            opts.penalty_scaling = 10
            opts.penalty_initial = 1000
        else
            opts.penalty_scaling = 10
            opts.penalty_initial = 10
        end
        opts.iterations = 10000
        opts.static_bp = false
        opts.square_root = true
    end

    # Create model.
    optim_model = PGS_model_obj(PG_samples, K, phi, n_x, n_u, v_vec)

    # Define diagonal quadratic objective.
    Q_cost_diag = [0; 0] # diagonal of Q_cost
    Q_cost = Diagonal(repeat(Q_cost_diag, outer=K))
    R_cost = Diagonal(R_cost_diag)
    stage_cost = DiagonalCost(Q_cost, R_cost)
    obj = Objective(stage_cost, H)

    # Create constraint list.
    cons = ConstraintList(n_x * K, n_u, H)

    # Extend vectors if necessary.
    if size(u_min, 2) == 1
        u_min = repeat(u_min, 1, H)
    end

    if size(u_max, 2) == 1
        u_max = repeat(u_max, 1, H)
    end

    if size(y_min, 2) == 1
        y_min = repeat(y_min, 1, H)
    end

    if size(y_max, 2) == 1
        y_max = repeat(y_max, 1, H)
    end

    # Define constraints for the input.
    for t in 1:H
        if !all(isinf.(u_min[:, t])) || !all(isinf.(u_max[:, t])) # ignore constraints that are set to infinity
            u_bnd_temp = BoundConstraint(n_x * K, n_u; u_min=u_min[:, t], u_max=u_max[:, t])
            add_constraint!(cons, u_bnd_temp, t)
        end
    end

    # Define constraints for the output.
    # TrajectoryOptimization.jl currently currently only supports constraints on x_vec. 
    # In this script, the simple measurement function y_t^k = x_t^k[1 : n_y] + e_t^k is assumed (i.e., the first n_y states are measured with noise), and the constraints are formulated as 
    # x_t^k[1 : n_y] >= y_{min,t} - e_t^k
    # x_t^k[1 : n_y] <= y_{max,t} - e_t^k.
    # The right-hand side of the inequalities are stored in x_vec_min_temp and x_vec_max_temp, respectively. For the states x_t^k[n_y + 1 : n_x] the max/min values are set to infinity.
    # Other measurement functions would have to be implemented by adding additional states representing the outputs.
    for t in 1:H
        if !all(isinf.(y_min[:, t])) || !all(isinf.(y_max[:, t]))
            # Initialize vectors that contain the minimum and maximum values for x_vec.
            x_max_temp = Array{Float64}(undef, n_x * K)
            x_min_temp = Array{Float64}(undef, n_x * K)
            for k in 1:K
                # Only consider the constraints for the models in active_constraints.
                if k in active_constraints
                    x_max_temp[n_x*(k-1)+1:n_x*k] = [y_max[:, t] - e_vec[:, t, k]; fill(Inf, n_x - n_y)]
                    x_min_temp[n_x*(k-1)+1:n_x*k] = [y_min[:, t] - e_vec[:, t, k]; fill(-Inf, n_x - n_y)]
                else
                    x_max_temp[n_x*(k-1)+1:n_x*k] = fill(Inf, n_x)
                    x_min_temp[n_x*(k-1)+1:n_x*k] = fill(-Inf, n_x)
                end
            end
            # Add the constraints to the constraint list.
            x_bnd_temp = BoundConstraint(n_x * K, n_u; x_min=x_min_temp, x_max=x_max_temp)
            add_constraint!(cons, x_bnd_temp, t)
        end
    end

    # Create problem.
    prob = Problem(optim_model, obj, x_vec_0, H - 1.0, constraints=cons, dt=1.0)

    # Roll out initial u.
    if !(u_init === nothing)
        initial_controls!(prob, u_init)
        rollout!(prob)
    end

    # Create ALTROSolver.
    solver = ALTROSolver(prob, opts, show_summary=print_progress)

    if print_progress
        println("### Started optimization algorithm")
    end

    # Solve the OCP.
    solve!(solver)

    time_optimization = time() - optimization_timer

    if print_progress
        @printf("### Optimization complete\nRuntime: %.2f s\n", time_optimization)
    end

    # Extract the solution and convert to matrices - copy is used to avoid problems with the garbage collection.
    x_opt = copy(permutedims(reshape(hcat(Vector.(states(solver))...), (n_x, K, H)), (1, 3, 2)))
    u_opt = copy(reshape(hcat(Vector.(controls(solver))...), (n_u, H - 1)))
    J_opt = copy(cost(solver))

    # Get the output via the measurement function y_t^k = x_t^k[1 : n_y] + e_t^k.
    y_opt = Array{Float64}(undef, n_y, H, K)
    for t in 1:H
        for k in 1:K
            y_opt[:, t, k] = x_opt[1:n_y, t, k] + e_vec[:, t, k]
        end
    end

    return x_opt, u_opt, y_opt, J_opt, copy(Integer(status(solver))), copy(solver.stats.iterations), copy(solver.stats.iterations_outer), copy(solver.stats.penalty_max)
end

"""
    epsilon(s::Int64, K::Int64, β::Float64)

Determine the parameter ϵ. 1-ϵ correponds to a bound on the probability that the incurred cost exceeds the worst-case cost or that the constraints are violated when the input trajectory u_{0:H} is applied to the unknown system.
ϵ is the unique solution over the inverval (0,1) of the polynomial equation in the v variable:
binomial(K, s) * (1 - v)^(K - s) - (β / K) * ∑_{m = s}^{K - 1} binomial(m, s) * (1 - v)^(m - s) = 0

# Arguments
- `s`: cardinality of the support sub-sample 
- `K`: number of scenarios
- `β`: confidence parameter

This function is based on the paper
    S. Garatti and M. C. Campi, “Risk and complexity in scenario optimization,” Mathematical Programming, vol. 191, no. 1, pp. 243–279, 2022.
and the code provided in the appendix.
"""
function epsilon(s::Int64, K::Int64, β::Float64)
    alphaU = beta_inc_inv(K - s + 1, s, β)[2]
    m1 = Array(s:K)'
    aux1 = sum(triu(log.(ones(K - s + 1) * m1), 1), dims=2)
    aux2 = sum(triu(log.(ones(K - s + 1) * (m1 .- s)), 1), dims=2)
    coeffs1 = aux2 - aux1
    m2 = Array(K+1:4*K)'
    aux3 = sum(tril(log.(ones(3 * K) .* m2)), dims=2)
    aux4 = sum(tril(log.(ones(3 * K) .* (m2 .- s))), dims=2)
    coeffs2 = aux3 - aux4
    t1 = 0
    t2 = 1 - alphaU
    poly1 = 1 + β / (2 * K) - β / (2 * K) * sum(exp.(coeffs1 .- (K .- m1') * log(t1))) - β / (6 * K) * sum(exp.(coeffs2 .+ (m2' .- K) * log(t1)))
    poly2 = 1 + β / (2 * K) - β / (2 * K) * sum(exp.(coeffs1 .- (K .- m1') * log(t2))) - β / (6 * K) * sum(exp.(coeffs2 .+ (m2' .- K) * log(t2)))
    if !(poly1 * poly2 > 0)
        while t2 - t1 > 1e-10
            t = (t1 + t2) / 2
            polyt = 1 + β / (2 * K) - β / (2 * K) * sum(exp.(coeffs1 .- (K .- m1') * log(t))) - β / (6 * K) * sum(exp.(coeffs2 .+ (m2' .- K) * log(t)))
            if polyt > 0
                t2 = t
            else
                t1 = t
            end
        end
        ϵ = t1
    end
    return ϵ
end

"""
    solve_PG_OCP_greedy_guarantees(PG_samples::Vector{PG_sample}, phi::Function, R, H, u_min, u_max, y_min, y_max, R_cost_diag, β; x_vec_0=nothing, v_vec=nothing, e_vec=nothing, u_init=nothing, K_pre_solve=0, opts=nothing, print_progress=true)

Solve the following optimal control problem and determine a support sub-sample with cardinality s via a greedy constraint removal. 
Based on the cardinality s, a bound on the probability that the incurred cost exceeds the worst-case cost or that the constraints are violated when the input trajectory u_{0:H} is applied to the unknown system is calculated.

min ∑_{∀t} 1/2 * u_t * Diagonal(R_cost_diag) * u_t

subject to: ∀k, ∀t
    x_{t+1}^k = f(x_t^k, u_t^k) + v_t^k - implemented in RobotDynamics.discrete_dynamics
    x_t^k[1:n_y] >= y_{min,t} - e_t^k
    x_t^k[1:n_y] <= y_{max,t} - e_t^k
    u_t >= u_{min,t}
    u_t <= u_{max,t}.

Note that the output constraints imply the measurement function y_t^k = x_t^k[1:n_y].
Further note that the states of the individual models x^k are combined in the vector x_vec of dimension K*n_x.

# Arguments
- `PG_samples`: PG samples
- `phi`: basis functions
- `R`: variance of zero-mean Gaussian measurement noise - only used if e_vec is not passed
- `H`: horizon of the OCP
- `u_min`: array of dimension 1 x 1 or 1 x H containing the minimum control input for all timesteps
- `u_max`: array of dimension 1 x 1 or 1 x H containing the maximum control input for all timesteps
- `y_min`: array of dimension 1 x 1 or 1 x H containing the minimum system output for all timesteps
- `y_max`: array of dimension 1 x 1 or 1 x H containing the maximum system output for all timesteps
- `R_cost_diag`: parameter of the diagonal quadratic cost function
- `β`: confidence parameter
- `x_vec_0`: vector with K*n_x elements containing the initial state of all models - if not provided, the initial states are sampled based on the PGS samples
- `v_vec`: array of dimension n_x x H x K that contains the process noise for all models and all timesteps - if not provided, the noise is sampled based on the PGS samples
- `e_vec`: array of dimension n_y x H x K that contains the measurement noise for all models and all timesteps - if not provided, the noise is sampled based on the provided R
- `u_init`: initial guess for the optimal trajectory
- `K_pre_solve`: if K_pre_solve > 0, an initial guess for the optimal trajectory is obtained by solving the OCP with only K_pre_solve < K models
- `opts`: SolverOptions struct containing options of the solver
- `print_progress`: if set to true, the progress is printed
"""
function solve_PG_OCP_greedy_guarantees(PG_samples::Vector{PG_sample}, phi::Function, R, H, u_min, u_max, y_min, y_max, R_cost_diag, β; x_vec_0=nothing, v_vec=nothing, e_vec=nothing, u_init=nothing, K_pre_solve=0, opts=nothing, print_progress=true)
    # Time first optimization.
    first_solve_timer = time()

    # Get number of states, etc.
    n_x = size(PG_samples[1].x_m1, 1)
    n_u = size(u_min, 1)
    n_y = size(y_min, 1)
    K = size(PG_samples, 1)

    # Set solver options if not provided - see Altro.jl documentation.
    if opts === nothing
        opts = SolverOptions()
        opts.constraint_tolerance = 1e-5
        opts.cost_tolerance = 1e-3
        opts.cost_tolerance_intermediate = 10 * opts.cost_tolerance
        opts.projected_newton_tolerance = sqrt(opts.constraint_tolerance)
        opts.penalty_scaling = 10
        opts.penalty_initial = 10
        opts.iterations = 5000
        opts.max_cost_value = 1e10
        opts.iterations_outer = 30
        opts.static_bp = false
        opts.square_root = true
    end

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

    u_init, max_penalty = solve_PG_OCP(PG_samples, phi, R, H, u_min, u_max, y_min, y_max, R_cost_diag; x_vec_0=x_vec_0, v_vec=v_vec, e_vec=e_vec, u_init=u_init, K_pre_solve=K_pre_solve, opts=opts, print_progress=print_progress)[[2, 8]]

    # The OCP is solved again with initialization to obtain the optimal input u_opt.
    opts.iterations = 1000
    opts.penalty_initial = max_penalty[end] # increase initial penalty
    println("### Startet optimization of fully constrained problem with initialization")
    x_opt, u_opt, y_opt, J_opt, termination_status, iterations, iterations_outer = solve_PG_OCP(PG_samples, phi, R, H, u_min, u_max, y_min, y_max, R_cost_diag; x_vec_0=x_vec_0, v_vec=v_vec, e_vec=e_vec, u_init=u_init, K_pre_solve=0, opts=opts, print_progress=print_progress)[1:end-1]

    # Determine guarantees.
    # If a feasible u_opt is found, probabilistic constraint satisfaction guarantees are derived by greedily removing constraints to determine a support sub-sample S.
    if termination_status == 2
        time_first_solve = time() - first_solve_timer

        # Time computation of guarantees.
        guarantees_timer = time()

        # Determine support sub-samples and guarantees for the generalization of the resulting input trajectory.
        println("### Started search for support sub-sample")

        # Reduce number of iterations - if the number of iterations of the original OCP is exceeded, the solution will likely be different, and the optimization can be stopped.
        opts.iterations = ceil(Int, 2 * iterations)
        opts.iterations_outer = ceil(Int, 2 * iterations_outer)

        # Sort scenarios according to the distance to the constraint boundary - removing the scenarios with the largest distance to the constraint boundary first usually yields better results.
        dist = Array{Float64}(undef, K) # minimum distance of the scenarios to the constraint boundary
        for i in 1:K
            dist[i] = minimum([minimum([norm((y_opt[:, :, i]-y_min)[:, j]) for j = 1:H]) minimum([norm((y_opt[:, :, i]-y_max)[:, j]) for j = 1:H])])
        end
        scenarios_sorted = sortperm(dist; rev=true) # sort scenarios

        # Pre-allocate.
        u_opt_temp = Array{Float64}(undef, n_u, H - 1)
        active_scenarios = scenarios_sorted

        # Greedily remove constraints and check, whether the solution changes to determine a support sub-sample.
        # In this implementation, the dynamic constraints are always taken into account for all scenarios and only the output constraints are only taken into account for the scenarios in the set temp_scenarios. 
        # Since the dynamic constraints are trivially fulfilled, this does not change the optimization problem. 
        # This implementation ensures that the solutions of the solver are exactly the same if the constraints are fulfilled. 
        # Otherwise, there will be small deviations in the solution due to the numerics of the solver and a threshold must be used in line 519.
        for i in 1:K
            # Print progress.
            @printf("Startet optimization with new constraint set\nIteration: %i/%i\n", i, K)

            # Temporaily remove the constraints corresponding to the PG samples with index i from the constraint set.
            temp_scenarios = active_scenarios[active_scenarios.!==scenarios_sorted[i]]

            # Catch errors that might occur during the optimization, e.g., out-of-memory errors.
            try
                # Solve the OCP with reduced constraint set.
                u_opt_temp, termination_status_temp = solve_PG_OCP(PG_samples, phi, R, H, u_min, u_max, y_min, y_max, R_cost_diag; x_vec_0=x_vec_0, v_vec=v_vec, e_vec=e_vec, u_init=u_init, K_pre_solve=0, active_constraints=temp_scenarios, opts=opts, print_progress=print_progress)[[2, 5]]

                # If the optimization is successful and the solution does not change permanently, remove the constraints corresponding to the PG samples with index i from the constraint set.
                if ((termination_status_temp == 2) && (maximum(abs.(u_opt_temp - u_opt)) == 0))
                    active_scenarios = temp_scenarios
                end
            catch
                # If an error occurs, proceed with the next candidate for a support sub-sample.
                @warn "Optimization with temporarily removed constraints failed. Proceeding with next candidate for a support sub-sample."
                num_failed_optimizations += 1
            end
        end

        # Determine the cardinality of the support sub-sample.
        s = length(active_scenarios)

        # Based on the cardinality of the support sub-sample, determine the parameter ϵ. 
        # 1-ϵ correponds to a bound on the probability that the incurred cost exceeds the worst-case cost or that the constraints are violated when the input trajectory u_{0:H} is applied to the unknown system.
        epsilon_prob = epsilon(s, K, β)
        epsilon_perc = epsilon_prob * 100

        # Print s, ϵ, and runtime.
        time_guarantees = time() - guarantees_timer
        @printf("### Support sub sample found\nCardinality of the support sub-sample (s): %i\nMax. constraint violation probability (1-epsilon): %.2f %%\nTime to compute u*: %.2f s\nTime to compute 1-epsilon: %.2f s\n", s, 100-epsilon_perc, time_first_solve, time_guarantees)
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
    return x_opt, u_opt, y_opt, J_opt, s, epsilon_prob, epsilon_perc, time_first_solve, time_guarantees, num_failed_optimizations
end