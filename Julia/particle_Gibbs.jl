using Distributions
using LinearAlgebra
using StaticArrays
using Plots
using Printf
using Random
using StatsBase

# Struct for the samples of the PGS algorithm
mutable struct PG_sample
    A::Matrix{Float64} # weight matrix
    Q::Matrix{Float64} # process noise covariance
    w_m1::Array{Float64} # weights in the last timestep of the training dataset (t=-1)
    x_m1::Array{Float64} # corresponding states in the last timestep of the training dataset (t=-1)
    u_m1::Array{Float64} # input in the last timestep of the training dataset (t=-1) - required to make predictions
end

"""
    systematic_resampling(W, N)

Sample N indices according to the weights W.

# Arguments
- `W`: vector containing the weights of the particles
- `N`: number of indices to be sampled

This function is based on the paper
    A. Svensson and T. B. Schön, “A flexible state–space model for learning nonlinear dynamical systems,” Automatica, vol. 80, pp. 189–199, 2017.
and the code provided in the supplementary material.
"""
function systematic_resampling(W, N)
    # Normalize weights.
    W = W / sum(W)

    u = 1 / N * rand()
    idx = Array{Int}(undef, N, 1) # array containing the sampled indices
    q = 0
    n = 0
    for i in 1:N
        while q < u
            n = n + 1
            q = q + W[n]
        end
        idx[i] = n
        u = u + 1 / N
    end
    return idx
end

"""
    MNIW_sample(Phi, Psi, Sigma, V, Lambda_Q, ell_Q, T)

Sample new model parameters (A, Q) from the conditional distribution p(A, Q | x_T:-1), which is a matrix normal inverse Wishart (MNIW) distribution.

# Arguments
- `Phi`: statistic; see paper below for definition
- `Psi`: statistic; see paper below for definition
- `Sigma`: statistic; see paper below for definition
- `V`: left covariance matrix of MN prior on A
- `Lambda_Q`: scale matrix of IW prior on Q
- `ell_Q`: degrees of freedom of IW prior on Q
- `T`: length of the training trajectory

This function is based on the paper
    A. Svensson and T. B. Schön, “A flexible state–space model for learning nonlinear dynamical systems,” Automatica, vol. 80, pp. 189–199, 2017.
and the code provided in the supplementary material.
"""
function MNIW_sample(Phi, Psi, Sigma, V, Lambda_Q, ell_Q, T)
    n_x = size(Phi, 1) # number of states

    # Update statistics (only relevant if mean matrix M of MN prior on A is not zero).
    Phibar = Phi # + (M/V)*M'
    Psibar = Psi # +  M/V

    # Calculate the components of the posterior MNIW distribution over model parameters.
    Sigbar = Sigma + I / V
    cov_M = Lambda_Q + Phibar - (Psibar / Sigbar) * Psibar' # scale matrix of posterior IW distribution
    cov_M_sym = 0.5 * (cov_M + cov_M') # to ensure matrix is symmetric
    post_mean = Psibar / Sigbar # mean matrix of posterior MN distribution
    left_cov = I / Sigbar # left covariance matrix of posterior MN distribution
    left_cov_sym = 0.5 * (left_cov + left_cov') # to ensure matrix is symmetric

    # Sample Q from posterior IW distribution.
    iw = InverseWishart(ell_Q + T * n_x, cov_M_sym)
    Q = rand(iw)

    # Sample A from posterior MN distribution.
    mn = MatrixNormal(post_mean, Q, left_cov_sym)
    A = rand(mn)

    # Return model parameters.
    return A, Q
end

"""
    particle_Gibbs(u_training, y_training, K, K_b, k_d, N, phi::Function, Lambda_Q, ell_Q, Q_init, V, A_init, x_init_dist, g, R)

Run particle Gibbs sampler with ancestor sampling to obtain samples (A, Q, x_T:-1) from the joint parameter and state posterior distribution p(A, Q, x_T:-1 | D=(u_training, y_training)).

# Arguments
- `u_training`: training input trajectory
- `y_training`: training output trajectory
- `K`: number of models/scenarios to be sampled
- `K_b': length of the burn in period
- `k_d`: number of models/scenarios to be skipped to decrease correlation (thinning)
- `N`: number of particles
- `phi': basis functions
- `Lambda_Q`: scale matrix of IW prior on Q
- `ell_Q`: degrees of freedom of IW prior on Q
- `Q_init`: initial value of Q
- `V`: left covariance matrix of MN prior on A
- `A_init`: initial value of A
- `x_init_dist`: distribution of the initial state
- `g`: observation function
- `R`: variance of zero-mean Gaussian measurement noise
- `x_prim`: prespecified trajectory for the first iteration

This function is based on the papers
    A. Svensson and T. B. Schön, “A flexible state–space model for learning nonlinear dynamical systems,” Automatica, vol. 80, pp. 189–199, 2017.
    F. Lindsten, T. B. Schön, and M. Jordan, “Ancestor sampling for particle Gibbs,” Advances in Neural Information Processing Systems, vol. 25, 2012.
and the code provided in the supplementary material.
"""
function particle_Gibbs(u_training, y_training, K, K_b, k_d, N, phi::Function, Lambda_Q, ell_Q, Q_init, V, A_init, x_init_dist, g, R; x_prim = nothing)
    # Total number of models
    K_total = K_b + 1 + (K - 1) * (k_d + 1)

    # Get number of states, etc.
    n_x = size(A_init, 1)
    n_u = size(u_training, 1)
    n_y = size(y_training, 1)
    T = size(y_training, 2)

    # Define the prespecified trajectory for the first iteration if not provided.
    if x_prim === nothing
        x_prim = zeros(n_x, 1, T)
    end

    # Initialize.
    PG_samples = Vector{PG_sample}(undef, K)
    for k in 1:K
        PG_samples[k] = PG_sample(Array{Float64}(undef, size(A_init)), Array{Float64}(undef, size(Q_init)), Array{Float64}(undef, N), Array{Float64}(undef, n_x, N), Array{Float64}(undef, n_u))
    end
    w = Array{Float64}(undef, T, N)
    x_pf = Array{Float64}(undef, n_x, N, T)
    a = Array{Int64}(undef, T, N)
    current_sample = 1
    A = A_init
    Q = Q_init

    # Time PGS sampling.
    learning_timer = time()

    println("### Started learning algorithm")

    for k in 1:K_total
        # Get current model.
        f(x, u) = A * phi(x, u)
        mvn_v = MvNormal(zeros(n_x), Q) # process noise distribution

        # Initialize particle filter with ancestor sampling.
        x_pf[:, end, :] = x_prim

        # Sample initial states.
        for n in 1:N-1
            x_pf[:, n, 1] = rand(x_init_dist)
        end

        # Particle filter (resampling, propagation, and ancestor sampling)
        for t in 1:T
            if t >= 2
                if k > 1 # Run the conditional PF with ancestor sampling.
                    # Resample particles (= sample ancestors).
                    a[t, 1:N-1] = systematic_resampling(w[t-1, :], N - 1)

                    # Propagate resampled particles.
                    x_pf[:, 1:N-1, t] = f(x_pf[:, a[t, 1:N-1], t-1], repeat(u_training[:, t-1], 1, N - 1)) + rand(mvn_v, N - 1)

                    # Sample ancestors of prespecified trajectory x_prim.
                    mvn_x_prim = MvNormal(x_pf[:, N, t], Q)
                    waN = w[t-1, :] .* pdf(mvn_x_prim, f(x_pf[:, :, t-1], repeat(u_training[:, t-1], 1, N)))
                    waN = waN ./ sum(waN)
                    a[t, N] = systematic_resampling(waN, 1)[1]

                else # Run a standard PF on the first iteration.
                    # Resample particles.
                    a[t, :] = systematic_resampling(w[t-1, :], N)

                    # Propagate resampled particles.
                    x_pf[:, :, t] = f(x_pf[:, a[t, :], t-1], repeat(u_training[:, t-1], 1, N)) + rand(mvn_v, N)
                end
            end

            # PF weight update based on measurement model (logarithms are used for numerical reasons).
            if n_y == 1 # scalar output
                log_w = -(g(x_pf[:, :, t], repeat(u_training[:, t], 1, N)) .- y_training[1, t]) .^ 2 / 2 / R # logarithm of normal distribution pdf (ignoring scaling factors)
            else # vector-valued output
                log_w = -sum((y_training[:, t] .- g(x_pf[:, :, t], repeat(u_training[:, t], 1, N))) .* (R \ (y_training[:, t] .- g(x_pf[:, :, t], repeat(u_training[:, t], 1, N)))), dims=1) / 2 # logarithm of multivariate normal distribution pdf (ignoring scaling factors)
            end
            w[t, :] = exp.(log_w .- maximum(log_w))
            w[t, :] = w[t, :] ./ sum(w[t, :])
        end

        # Use sample if the burn-in period is reached and the sample is not removed by thinning.
        if K_b < k && (mod(k - (K_b + 1), k_d + 1) == 0)
            PG_samples[current_sample].A = A
            PG_samples[current_sample].Q = Q
            PG_samples[current_sample].w_m1 = w[end, :]
            PG_samples[current_sample].x_m1 = x_pf[:, :, end]
            PG_samples[current_sample].u_m1 = u_training[:, end]
            current_sample += 1
        end

        # Sample state trajectory x_T:-1 to condition on.
        star = systematic_resampling(w[end, :], 1)
        x_prim[:, 1, T] = x_pf[:, star, T]
        for t in T-1:-1:1
            star = a[t+1, star]
            x_prim[:, 1, t] = x_pf[:, star, t]
        end

        # Sample new model parameters conditional on sampled trajectory, i.e., sample from p(A, Q | x_T:-1).
        zeta = x_prim[:, 1, 2:T]
        z = phi(x_prim[:, 1, 1:T-1], u_training[:, 1:T-1])
        Phi = zeta * zeta' # statistic; see paper "A flexible state-space model for learning nonlinear dynamical systems"
        Psi = zeta * z' # statistic
        Sigma = z * z' # statistic
        A, Q = MNIW_sample(Phi, Psi, Sigma, V, Lambda_Q, ell_Q, T - 1) # sample new model parameters

        # Print progress.
        @printf("Iteration %i/%i\n", k, K_total)
    end
    # Print runtime.
    time_learning = time() - learning_timer
    @printf("### Learning complete\nRuntime: %.2f s\n", time_learning)

    return PG_samples
end

"""
    test_prediction(PG_samples::Vector{PG_sample}, phi::Function, g, R, k_n, u_test, y_test)

Simulate the PGS samples forward in time and compare the predictions to the test data.

# Arguments
- `PG_samples`: PG samples
- `phi`: basis functions
- `g`: observation function
- `R`: variance of zero-mean Gaussian measurement noise
- `k_n`: each model is simulated k_n times
- `u_test`: test input
- `y_test`: test output
"""
function test_prediction(PG_samples::Vector{PG_sample}, phi::Function, g, R, k_n, u_test, y_test)
    println("### Testing model")

    # Get number of models, etc.
    K = size(PG_samples, 1)
    n_x = size(PG_samples[1].A, 1)
    n_y = size(y_test, 1)
    T_test = size(y_test, 2)

    # Measurement noise distribution
    mvn_e = MvNormal(zeros(n_y), R)

    # Pre-allocate.
    x_test_sim = Array{Float64}(undef, n_x, T_test + 1, K, k_n)
    y_test_sim = Array{Float64}(undef, n_y, T_test, K, k_n)

    # Simulate models forward.
    Threads.@threads for k in 1:K
        # Get current model.
        A = PG_samples[k].A
        Q = PG_samples[k].Q
        f(x, u) = A * phi(x, u)
        mvn_v = MvNormal(zeros(n_x), Q) # process noise distribution

        # Simulate each model k_n times.
        for kn in 1:k_n
            # Pre-allocate
            x_loop = Array{Float64}(undef, n_x, T_test + 1)
            y_loop = Array{Float64}(undef, n_y, T_test)

            # Sample initial state.
            star = systematic_resampling(PG_samples[k].w_m1, 1)
            x_m1 = PG_samples[k].x_m1[:, star]
            x_loop[:, 1] = f(x_m1, PG_samples[k].u_m1) + rand(mvn_v)

            # Simulate model forward.
            for t in 1:T_test
                if t >= 2
                    x_loop[:, t] = f(x_loop[:, t-1], u_test[:, t-1]) + rand(mvn_v)
                end
                y_loop[:, t] = g(x_loop[:, t], u_test[:, t]) + rand(mvn_e)
            end

            # Store trajectory.
            x_test_sim[:, :, k, kn] = x_loop
            y_test_sim[:, :, k, kn] = y_loop
        end
    end

    # Reshape.
    x_test_sim = reshape(x_test_sim, (n_x, T_test + 1, K * k_n))
    y_test_sim = reshape(y_test_sim, (n_y, T_test, K * k_n))

    println("### Testing complete")

    # Plot results.
    plot_predictions(y_test_sim, y_test; plot_percentiles=true)

    # Compute and print RMSE.
    mean_rmse = sqrt(mean((y_test_sim .- repeat(y_test', 1, 1, K * k_n)) .^ 2))
    @printf("Mean rmse: %.2f\n", mean_rmse)
end

"""
    plot_predictions(y_pred, y_test; plot_percentiles=false, y_min=nothing, y_max=nothing)

Plot the predictions and the test data.

# Arguments
- `y_pred`: matrix containing the output predictions
- `y_test`: test output trajectory
- `plot_percentiles`: if set to true, percentiles are plotted
- `y_min`: min output to be plotted as constraint
- `y_max`: max output to be plotted as constraint
"""
function plot_predictions(y_pred, y_test; plot_percentiles=false, y_min=nothing, y_max=nothing)
    # Get prediction horizon and number of outputs.
    T_pred = size(y_test, 2)
    n_y = size(y_pred, 1)

    # Plot the predictions and the test data for all output dimensions.
    for i = 1:n_y
        # Calculate median, mean, maximum, and minimum prediction.
        y_pred_med = median(y_pred, dims=3)[i, :, 1]
        y_pred_mean = mean(y_pred, dims=3)[i, :, 1]

        y_pred_max = maximum(y_pred, dims=3)[i, :, 1]
        y_pred_min = minimum(y_pred, dims=3)[i, :, 1]

        # Calculate percentiles.
        y_pred_09 = mapslices(x -> quantile(x, 0.9), y_pred, dims=3)[i, :, 1]
        y_pred_01 = mapslices(x -> quantile(x, 0.1), y_pred, dims=3)[i, :, 1]

        # Plot range of predictions.
        p = plot(Array(0:T_pred-1), y_pred_min, fillrange=y_pred_max, alpha=0.35, label="all predictions", legend=:topleft)

        # Plot percentiles.
        if plot_percentiles
            plot!(Array(0:T_pred-1), y_pred_01, fillrange=y_pred_09, alpha=0.35, label="10% perc. - 90% perc.")
        end

        # Plot true output.
        plot!(Array(0:T_pred-1), y_test[i, :], label="true output", lw=2)

        # Plot median/mean prediction.
        # plot!(Array(0:T_pred-1), y_pred_med, label="median prediction", lw=2)
        plot!(Array(0:T_pred-1), y_pred_mean, label="mean prediction", lw=2)

        # Plot constraints.
        if y_min !== nothing
            plot!(Array(0:T_pred-1), y_min', fillrange=minimum([y_pred_min; y_test']) * ones(T_pred), fillcolor=:red, alpha=0.35, label="constraints", legend=:topleft)
        end
        if y_max !== nothing
            plot!(Array(0:T_pred-1), y_max', fillrange=maximum([y_pred_max; y_test']) * ones(T_pred), fillcolor=:red, alpha=0.35, label="constraints")
        end

        if 1 < n_y
            title!("y_" * string(i) * ": predicted output vs. true output")
            ylabel!("y_" * string(i))
        else
            title!("predicted output vs. true output")
            ylabel!("y")
        end
        xlabel!("t")
        display(p)
    end
end

"""
    plot_autocorrelation(PG_samples::Vector{PG_sample}; max_lag=0)

Plot the autocorrelation function (ACF) of the PG samples. This might be helpful when adjusting the thinning parameter k_d.

# Arguments
- `PG_samples`: PG samples
- `max_lag`: maximum lag at which to calculate the ACF
"""
function plot_autocorrelation(PG_samples::Vector{PG_sample}; max_lag=0)
    # Get number of models.
    K = size(PG_samples, 1)

    # Get number of parameters of the PG samples.
    number_of_variables = length(PG_samples[1].A) + length(PG_samples[1].Q) + size(PG_samples[1].x_m1, 1)

    if max_lag == 0
        max_lag = K - 1
    end

    # Fill matrix with the series of the parameters of the PG samples.
    signal_matrix = Array{Float64}(undef, K, number_of_variables)
    for i in 1:K
        # Sample initial state.
        star = systematic_resampling(PG_samples[i].w_m1, 1)
        x_m1 = PG_samples[i].x_m1[:, star]
        signal_matrix[i, :] = [vec(PG_samples[i].A); vec(PG_samples[i].Q); vec(x_m1)]
    end

    # Calculate the autocorrelation.
    autocorrelation = autocor(signal_matrix, Array(0:max_lag); demean=true)

    # Plot the ACF.
    p = plot(yticks=-1:0.1:1)
    for i in 1:number_of_variables
        # Plot the ACF of the elements of A.
        if i == 1
            plot!(Array(0:max_lag), autocorrelation[:, i], lc=:red, lw=2, label="A")
        elseif 1 < i <= length(PG_samples[1].A)
            plot!(Array(0:max_lag), autocorrelation[:, i], lc=:red, lw=2, label="")
        # Plot the ACF of the elements of Q.  
        elseif i == length(PG_samples[1].A) + 1
            plot!(Array(0:max_lag), autocorrelation[:, i], lc=:blue, lw=2, label="Q")
        elseif (length(PG_samples[1].A) + 1 < i) && (i <= length(PG_samples[1].A) + length(PG_samples[1].Q))
            plot!(Array(0:max_lag), autocorrelation[:, i], lc=:blue, lw=2, label="")
        # Plot the ACF of the elements of x_t-1.
        elseif i == length(PG_samples[1].A) + length(PG_samples[1].Q) + 1
            plot!(Array(0:max_lag), autocorrelation[:, i], lc=:green, lw=2, label="x")
        elseif length(PG_samples[1].A) + length(PG_samples[1].Q) + 1 < i
            plot!(Array(0:max_lag), autocorrelation[:, i], lc=:green, lw=2, label="")
        end
    end

    title!("Autocorrelation Function (AFC)")
    xlabel!("Lag")
    ylabel!("AFC")
    display(p)
end