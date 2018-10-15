
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

### Type definition

The first parametric composite type is used to store model parameters.


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

The second composite type is used to store discretized state variables.


```julia
struct StateVars{T, dim1} # add dim2, dim3, ... if more than one state variable
    x::SVector{dim1, T}
    
    function StateVars(dim1; T = Float64)
        x = SVector{dim1, T}(range(zero(T), stop=one(T), length=dim1))
        return new{T, dim1}(x)
    end
end
```

Note that static arrays, instead of regular ones, are used for better performance. Static arrays conveniently keep the length as a type parameter, which will be repeatedly used later. There is only one state variable in this model whose value ranges from 0 to 1, but this composite type can be easily extended to accommodate more state variables with other ranges and discretization methods.

The third composite type is used to store everything including model solution.


```julia
struct ModelSol{T, dim1} # add dim2, dim3, ... if more than one state variable
    param::ModelParam{T}
    state::StateVars{T, dim1}
    sol::Dict{Symbol, StaticVector{dim1, T}}
    
    function ModelSol(param::ModelParam{T}, state::StateVars{T, dim1}) where {T, dim1}
        sol = Dict{Symbol, StaticVector{dim1, T}}()
        return new{T, dim1}(param, state, sol)
    end
end
```

The model solution is a set of equilibrium quantities represented by static arrays and stored in a dictionary. They provide (approximate) values of these quantities at every discretized state of the model economy.

### Function definition


```julia
function pde!(du, u, p, t)
    param, state = p
    
    γA = param.γA ; ψA = param.ψA ; 
    γB = param.γB ; ψB = param.ψB ; 
    ρ = param.ρ ; φ = param.φ ; νA = param.νA ; 
    μ = param.μ ; σ = param.σ ; 
    B1 = param.B1 ; δ1 = param.δ1 ; 
    B2 = param.B2 ; δ2 = param.δ2 ; ω = param.ω ;
    
    for i in 1:length(state.x)
        x = state.x[i]
        
        pA = u[i,1] ; pAx, pAxx = fntdiff1d(state.x, view(u,:,1), i);
        pB = u[i,2] ; pBx, pBxx = fntdiff1d(state.x, view(u,:,2), i);
        ϕ1 = u[i,3] ; ϕ1x, ϕ1xx = fntdiff1d(state.x, view(u,:,3), i);
        ϕ2 = u[i,4] ; ϕ2x, ϕ2xx = fntdiff1d(state.x, view(u,:,4), i);
        
        # volatility of X, pA, pB, ϕ1, ϕ2, CA, CB and market price of risk κ
        Γ = 1 / (x / γA + (1 - x) / γB)
        p = x * pA + (1 - x) * pB
        σx = σ * x * (Γ / γA - 1) / (1 + Γ * x * (1 - x) / (γA * γB) * ((1 - γB * ψB) / (ψB - 1) * (pBx / pB) - (1 - γA * ψA) / (ψA - 1) * (pAx / pA)))
        σpA = pAx / pA * σx
        σpB = pBx / pB * σx 
        σϕ1 = ϕ1x / ϕ1 * σx
        σϕ2 = ϕ2x / ϕ2 * σx
        κ = Γ * (σ - x * (1 - γA * ψA) / (γA * (ψA - 1)) * σpA - (1 - x) * (1 - γB * ψB) / (γB * (ψB - 1)) * σpB)
        σCA = κ / γA + (1 - γA * ψA) / (γA * (ψA - 1)) * σpA
        σCB = κ / γB + (1 - γB * ψB) / (γB * (ψB - 1)) * σpB
        
        # drift of X, pA, pB, ϕ1, ϕ2, CA, CB and interest rate r
        # A.16 Equation in Garleanu Panageas has a typo
        mcA = κ^2 * (1 + ψA) / (2 * γA) + (1 - ψA * γA) / (γA * (ψA - 1)) * κ * σpA - (1 - γA * ψA) / (2 * γA * (ψA - 1)) * σpA^2
        mcB = κ^2 * (1 + ψB) / (2 * γB) + (1 - ψB * γB) / (γB * (ψB - 1)) * κ * σpB - (1 - γB * ψB) / (2 * γB * (ψB - 1)) * σpB^2
        r =  ρ + 1 / (ψA * x  + ψB * (1 - x))  * (μ - x * mcA - (1 - x) * mcB - φ * ((νA / pA + (1 - νA) / pB) * (ϕ1 + ϕ2) - 1))
        μCA = ψA * (r - ρ) + mcA
        μCB = ψB * (r - ρ) + mcB
        μx = x * (μCA - φ - μ) + φ * νA / pA * (ϕ1 + ϕ2) - σ * σx  
        μpA = pAx / pA * μx + 0.5 * pAxx / pA * σx^2
        μpB = pBx / pB * μx + 0.5 * pBxx / pB * σx^2
        μϕ1 = ϕ1x / ϕ1 * μx + 0.5 * ϕ1xx / ϕ1 * σx^2
        μϕ2 = ϕ2x / ϕ2 * μx + 0.5 * ϕ2xx / ϕ2 * σx^2
        
        du[i,1] = pA * (1 / pA + (μCA - φ) + μpA + σCA * σpA - r - κ * (σpA + σCA))
        du[i,2] = pB * (1 / pB + (μCB - φ) + μpB + σCB * σpB - r - κ * (σpB + σCB))
        du[i,3] = ϕ1 * (B1 * ω / ϕ1 + (μ - φ - δ1) + μϕ1 + σ * σϕ1 - r - κ * (σϕ1 + σ))
        du[i,4] = ϕ2 * (B2 * ω / ϕ2 + (μ - φ - δ2) + μϕ2 + σ * σϕ2 - r - κ * (σϕ2 + σ))
    end    
end
```
