"Core module of the PGopt algorithm. Contains all necessary functions such as the particle Gibbs sampler and scenario optimization."
module PGopt
using Distributions
using LinearAlgebra
using StaticArrays
using Plots
using Printf
using Random
using StatsBase
using Altro
using TrajectoryOptimization
using RobotDynamics
using ForwardDiff
using FiniteDiff
using SpecialFunctions
using JuMP
using Ipopt


export PG_sample, particle_Gibbs, systematic_resampling, MNIW_sample, test_prediction, plot_predictions, plot_autocorrelation, epsilon, solve_PG_OCP_Altro, solve_PG_OCP_Altro_greedy_guarantees, solve_PG_OCP_Ipopt, solve_PG_OCP_Ipopt_greedy_guarantees

# Struct for the samples of the PGS algorithm
mutable struct PG_sample
    A::Matrix{Float64} # weight matrix
    Q::Matrix{Float64} # process noise covariance
    w_m1::Array{Float64} # weights in the last timestep of the training dataset (t=-1)
    x_m1::Array{Float64} # corresponding states in the last timestep of the training dataset (t=-1)
    u_m1::Array{Float64} # input in the last timestep of the training dataset (t=-1) - required to make predictions
end

include("particle_Gibbs.jl")
include("optimal_control_Altro.jl")
include("optimal_control_Ipopt.jl")
end