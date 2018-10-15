# Compute the finite difference approximation of ∂ₓy and ∂ₓₓy at node (i) on an even/uneven grid
function fntdiff1d(x::StaticVector{N,T}, y::AbstractArray{T,1}, i::Int64) where {T<:Union{Float64, BigFloat}, N}
    length(y) == N || throw(DimensionMismatch("x and y in fntdiff1d"))

    if 1 < i < N
        ∂ₓy = /( y[i+1] - y[i-1] , x[i+1] - x[i-1] )
        
        ∂ₓₓy = /(+ *(x[i]   - x[i-1] , y[i+1])
                 - *(x[i+1] - x[i-1] , y[i]  )
                 + *(x[i+1] - x[i]   , y[i-1])
                ,
                 *(1/2            ,
                   x[i+1] - x[i-1],
                   x[i+1] - x[i]  ,
                   x[i]   - x[i-1])
                )
    elseif 1 == i
        ∂ₓy = /( y[i+1] - y[i] , x[i+1] - x[i] )
        
        ∂ₓₓy = /(+ *(x[i+1] - x[i]   , y[i+2])
                 - *(x[i+2] - x[i]   , y[i+1])
                 + *(x[i+2] - x[i+1] , y[i]  )
                ,
                 *(1/2            ,
                   x[i+2] - x[i]  ,
                   x[i+2] - x[i+1],
                   x[i+1] - x[i]  )
                )
    elseif N == i
        ∂ₓy = /( y[i] - y[i-1] , x[i] - x[i-1] )
        
        ∂ₓₓy = /(+ *(x[i-1] - x[i-2] , y[i]  )
                 - *(x[i]   - x[i-2] , y[i-1])
                 + *(x[i]   - x[i-1] , y[i-2])
                ,
                 *(1/2            ,
                   x[i]   - x[i-2],
                   x[i]   - x[i-1],
                   x[i-1] - x[i-2])
                )
    else
        error("BoundsError: index $i ∉ [1, $N]")
    end
    
    return ∂ₓy, ∂ₓₓy
end 



# Compute the finite difference approximation of ∂ₓy and ∂ₓₓy at all nodes on an even/uneven grid
function fntdiff1d!(∂ₓy::AbstractArray{T,1}, ∂ₓₓy::AbstractArray{T,1}, x::StaticVector{N,T}, y::AbstractArray{T,1}) where {T<:Union{Float64, BigFloat}, N}
    length(y) == length(∂ₓy) == length(∂ₓₓy) == N || throw(DimensionMismatch("x, y, ∂ₓy, and ∂ₓₓy in fntdiff1d!"))
    
    for i in 1:N
        ∂ₓy[i], ∂ₓₓy[i] = fntdiff1d(x, y, i)
    end
    return nothing
end
