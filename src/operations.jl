
############# Tensor Algebra ################

@inline Base.:+(a::AbstractTensor) = a

@inline Base.:+(a::AbstractTensor{N}, b::AbstractTensor{N}) where {N} = Tensor(a.x + b.x, a.y + b.y, a.z + b.z)

@inline Base.:+(a::AbstractTensor{N}...) where {N} = Tensor(+(map(_x, a)...), +(map(_y, a)...), +(map(_z, a)...))

# We treat Vec's as scalar for broadcasting but the default definition of + and - for AbstractArray's relies
# on broadcasting to perform addition and subtraction. The method definitons below overcomes this inconsistency
@inline Base.:+(a::AbstractTensor{N}, b::AbstractArray{<:Any, N}) where {N} = Array(a) + b

@inline Base.:+(b::AbstractArray{<:Any, N}, a::AbstractTensor{N}) where {N} = b + Array(a)

@inline Base.:-(a::T) where {T <: AbstractTensor} = @inline constructor(T)(map(-, fields(a))...)

@inline Base.:-(a::AbstractTensor{N}, b::AbstractTensor{N}) where {N} = Tensor(a.x - b.x, a.y - b.y, a.z - b.z)

@inline Base.:-(a::AbstractTensor{N}, b::AbstractArray{<:Any, N}) where {N} = Array(a) - b

@inline Base.:-(b::AbstractArray{<:Any, N}, a::AbstractTensor{N}) where {N} = b - Array(a)

import Base: ==
@inline (==)(a::AbstractTensor{N}, b::AbstractTensor{N}) where {N} = ((a.x == b.x) & (a.y == b.y) & (a.z == b.z))

@inline Base.:*(b::Number, v::T) where {T <: AbstractTensor} = @inline  begin
    bt = _my_convert(promote_type(typeof(b), nonzero_eltype(T)), b)
    constructor(T)(map(*, ntuple(i -> bt, Val(fieldcount(T))), fields(v))...)
end

@inline Base.:*(v::AbstractTensor, b::Number) = b * v

@inline Base.muladd(b::Number, v::T, u::T) where {T <: AbstractTensor} = @inline  begin
    bt = _my_convert(promote_type(typeof(b), nonzero_eltype(T)), b)
    constructor(T)(map(muladd, ntuple(i -> bt, Val(fieldcount(T))), fields(v), fields(u))...)
end

@inline Base.muladd(a::Number, v::AbstractTensor{N}, u::AbstractTensor{N}) where {N} = Tensor(muladd(a, v.x, u.x), muladd(a, v.y, u.y), muladd(a, v.z, u.z))

@inline Base.muladd(::Zero, ::AbstractTensor{N}, u::AbstractTensor{N}) where {N} = u

@inline Base.muladd(::One, v::AbstractTensor{N}, u::AbstractTensor{N}) where {N} = v+u

@inline Base.muladd(v::AbstractTensor{N}, a::Number, u::AbstractTensor{N}) where {N} = muladd(a, v, u)

@inline Base.:/(v::T, b::Number) where {T <: AbstractTensor} = @inline  begin
    ib = inv(b)
    bt = _my_convert(promote_type(typeof(ib), nonzero_eltype(T)), ib)
    constructor(T)(map(*, fields(v), ntuple(i -> bt, Val(fieldcount(T))))...)
end

@inline Base.://(v::AbstractTensor, b::Number) = (One() // b) * v

@inline Base.zero(::Type{T}) where {T <: AbstractTensor} = @inline constructor(T)(_zero_for_tuple(fieldtypes(T)...)...)

@inline Base.zero(::T) where {T <: AbstractTensor} = zero(T)

@inline Base.conj(a::T) where {T <: AbstractTensor} = @inline constructor(T)(map(conj, fields(a))...)


################ Tensor operations ##########################


@inline dot(a::AbstractTensor,b::Vec) = muladd(a.x, b.x, muladd(a.y, b.y, a.z*b.z))

@inline dot(A::AbstractTensor, B::AbstractTensor) = Tensor(dot(A,B.x), dot(A, B.y), dot(A, B.z)) 

@inline Base.:*(T::Ten, B::Union{Ten,Vec}) = dot(T,B)

"""
    dotadd(x::AbstractTensor{N}, y::AbstractTensor{M}, z::Union{AbstractTensor{N + M - 2}, Nuber) -> Union{AbstractTensor{N + M - 2}, Number}

Equivalent to `dot(x, y) + z`, but uses combined multiply-add ([`muladd`](@ref)) on every computation for efficiency. See [`muladd`](@ref) and [`LinearAlgebra.dot`](@ref).
"""
@inline dotadd(a::Vec, b::Vec, c::Number) = muladd(a.x, b.x, muladd(a.y, b.y, muladd(a.z, b.z, c)))

@inline function dotadd(a::AbstractTensor{N1}, b::Vec, c::AbstractTensor{N2}) where {N1, N2}
    ((N1-1) === N2) || throw(DimensionMismatch())
    return muladd(a.x, b.x, muladd(a.y, b.y, muladd(a.z, b.z, c)))
end

@inline function dotadd(a::AbstractTensor{N1}, b::AbstractTensor{N2}, c::AbstractTensor{N3}) where {N1, N2, N3}
    ((N1+N2-2) === N3) || throw(DimensionMismatch())
    return Tensor(dotadd(a, b.x, c.x), dotadd(a, b.y, c.y), dotadd(a, b.z, c.z))
end

@inline Base.muladd(A::Ten, B::Vec, C::Vec) = dotadd(A, B, C)

@inline Base.muladd(A::Ten, B::Ten, C::Ten) = dotadd(A, B, C)

"""
    inner(a::AbstractTensor{N}, b::AbstractTensor{N}) -> Number

Compute the inner product between two `N`th order tensors. For complex tensors, the first tensor is conjugated.
"""
@inline inner(u::Vec, v::Vec) = dot(conj(u), v)

"""
    inneradd(a::AbstractTensor{N}, b::AbstractTensor{N}, c::Number) -> Number

Equivalent to `inner(a,b) + c`, but uses combined multiply-add ([`muladd`](@ref)) on every computation for efficiency. See [`muladd`](@ref) and [`inner`](@ref).
"""
@inline inneradd(u::Vec, v::Vec, c::Number) = dotadd(conj(u), v, c)

@inline inneradd(T1::AbstractTensor{N}, T2::AbstractTensor{N}, c::Number) where {N} = inneradd(T1.x, T2.x, inneradd(T1.y, T2.y, inneradd(T1.z, T2.z, c)))

@inline inner(T1::AbstractTensor{N}, T2::AbstractTensor{N}) where {N} = inneradd(T1.x, T2.x, inneradd(T1.y, T2.y, inner(T1.z,T2.z)))

"""
    dcontract(a::AbstractTensor{N>=2}, b::AbstractTensor{M>=2}) -> Union{AbstractTensor{N + M - 4}, Number}

Contracts (sums over the product of the elements) the two innermost indices of tensors `a` and `b`. The result is a tensor of order `N + M - 4`. If `N == M == 2` returns a `Number`.
The symbol `⊡`, writen `\\boxdot`, is overloaded for this operation. 
"""
@inline dcontract(a::AbstractTensor,b::Ten) = muladd(a.xx, b.xx, muladd(a.xy, b.xy, muladd(a.xz, b.xz,
                                              muladd(a.yx, b.yx, muladd(a.yy, b.yy, muladd(a.yz, b.yz,
                                              muladd(a.zx, b.zx, muladd(a.zy, b.zy, a.zz*b.zz))))))))

@inline dcontract(A::AbstractTensor, B::AbstractTensor) = Tensor(dcontract(A, B.x), dcontract(A, B.y), dcontract(A, B.z)) 

const ⊡ = dcontract

"""
    dcontractadd(a::AbstractTensor{N>=2}, b::AbstractTensor{M>=2}, c::Union{AbstractTensor{N + M - 4}, Number}) -> Union{AbstractTensor{N + M - 4}, Number}

Equivalent to `dcontract(a,b) + c`, but uses combined multiply-add ([`muladd`](@ref)) on every computation for efficiency. See [`muladd`](@ref) and [`dcontract`](@ref).
"""
@inline dcontractadd(a::Ten,b::Ten, c::Number) = muladd(a.xx, b.xx, muladd(a.xy, b.xy, muladd(a.xz, b.xz,
                                         muladd(a.yx, b.yx, muladd(a.yy, b.yy, muladd(a.yz, b.yz,
                                         muladd(a.zx, b.zx, muladd(a.zy, b.zy, muladd(a.zz, b.zz, c)))))))))

@inline function dcontractadd(a::AbstractTensor{N},b::Ten, c::AbstractTensor{N2}) where {N,N2}
    ((N-2) === N2) || throw(DimensionMismatch())
    return muladd(a.xx, b.xx, muladd(a.xy, b.xy, muladd(a.xz, b.xz,
           muladd(a.yx, b.yx, muladd(a.yy, b.yy, muladd(a.yz, b.yz,
           muladd(a.zx, b.zx, muladd(a.zy, b.zy, muladd(a.zz, b.zz, c)))))))))
end

@inline function dcontractadd(A::AbstractTensor{N}, B::AbstractTensor{N2}, C::AbstractTensor{N3}) where {N,N2,N3}
    ((N-2 + N2-2) === N3) || throw(DimensionMismatch())
    return Tensor(dcontractadd(A, B.x, C.x), dcontractadd(A, B.y, C.y), dcontractadd(A, B.z, C.z)) 
end

#We use internal `_otimes` so we can define `_otimes(::Any, ::Any)` and don't make `otimes` work with anything other then AbstractTensors
@inline _otimes(a,b) = a*b

@inline _otimes(a::AbstractTensor, b::AbstractTensor) = Tensor(_otimes(a, b.x), _otimes(a, b.y), _otimes(a, b.z))

"""
    otimes(a::AbstractTensor{N}, b::AbstractTensor{M}) -> AbstractTensor{N + M}

Computes the open (tensor) product between two tensors. The result is a tensor of order `N + M`.
The symbol `⊗`, writen `\\otimes`, is overloaded for this operation. 
"""
@inline otimes(a::AbstractTensor, b::AbstractTensor) = _otimes(a, b)

"""
    otimes(a::AbstractTensor{N}) -> AbstractTensor{N + N}

Equivalent to `otimes(a,a)`. If `N == 1`, returns a `SymmetricTensor{2}`.
"""
@inline otimes(a::AbstractTensor) = otimes(a, a)

@inline _otimesadd(a, b, c) = muladd(a, b, c)

@inline _otimesadd(a::AbstractTensor, b::AbstractTensor, c::AbstractTensor) = Tensor(_otimesadd(a, b.x, c.x), _otimesadd(a, b.y, c.y), _otimesadd(a, b.z, c.z))

"""
    otimesadd(a::AbstractTensor{N}, b::AbstractTensor{M}, c::AbstractTensor{N + M}) -> AbstractTensor{N + M}

Equivalent to `otimes(a,b) + c`, but uses combined multiply-add ([`muladd`](@ref)) on every computation for efficiency. See [`muladd`](@ref) and [`otimes`](@ref).
"""
@inline otimesadd(a::AbstractTensor, b::AbstractTensor, c::AbstractTensor) = _otimesadd(a, b, c)

const ⊗ = otimes
