
# A Replication Exercise of Garleanu and Panageas (2015)

In this notebook, I seek to solve the model developed by [Garleanu and Panageas (2015)](http://dx.doi.org/10.1086/680996) and replicate their main results (plots). I greatly benefited from a similar exercise done by [Matthieu Gomez](https://github.com/matthieugomez).

To get started, the following modules should be in place:


```julia
using ParDiffEqnSuite
using StaticArrays
using DifferentialEquations
using BenchmarkTools
using Plots
using Printf
using Distances
```

where `ParDiffEqnSuite.jl` is a user-defined module that can be found in `./src/`.

## Preparation

First, I define a parametric composite type to store model parameters.


```julia
struct ModelParam{T}
    # utility function
    γA::T
    ψA::T 
    γB::T 
    ψB::T
    ρ::T 
    φ::T

    # proportion a
    νA::T

    # consumption
    μ::T
    σ::T

    # earning function
    B1::T 
    δ1::T
    B2::T 
    δ2::T
    ω::T
        
    # suppress generation of default constructors
    function ModelParam(; T = Float64,
                        γA  = 1.5, ψA = 0.7, γB = 10.0, ψB = 0.05, 
                        ρ = 0.001, φ = 0.02, νA = 0.01, μ = 0.02, σ = 0.041, 
                        B1 = 30.72, δ1 = 0.0525, B2 = -30.29, δ2 = 0.0611, ω = 0.92)
        scale = φ / (φ + δ1) * B1 + φ / (φ + δ2) * B2
        B1 = B1 / scale
        B2 = B2 / scale
        return new{T}(γA , ψA, γB, ψB, ρ, φ, νA, μ, σ, B1, δ1, B2, δ2, ω)
    end
end
```

One can construct a new `ModelParam` object by calling `x = ModelParam()`, which will deploy the baseline parameter values in the paper. Or instead, one can deviate from the baseline by calling something like `x = ModelParam(ψA = 0.9, ψB = 0.5)` to change one or more paramter values while keeping the others the same. The default number type is `Float64`, but one can choose `T = BigFloat` for higher precision.

Then I define another composite type to store discretized state variables.


```julia
struct StateVars{T, dim1} # add dim2, dim3, ... if more than one state variable
    x::SVector{dim1, T}
    
    function StateVars(dim1; T = Float64)
        x = SVector{dim1, T}(range(zero(T), stop=one(T), length=dim1))
        return new{T, dim1}(x)
    end
end
```

Note that static arrays, instead of regular ones, are used for better performance. Static arrays conveniently keep the length as a type parameter, which will be repeatedly used later. There is only one state variable in this model, but this composite type can be easily extended to accomodate higher-dimension problems.
