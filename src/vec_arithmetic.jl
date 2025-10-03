import Base: /, //, muladd, conj, ==, zero

@inline dot(a::Number, b::Number) = a*b
include("muladd_definitions.jl")

# Definitions for abstract Tensors
@inline Base.:+(a::AbstractTensor) = a
@inline Base.:*(b::Number, v::T) where {T <: AbstractTensor} = @inline  begin
    bt = convert(promote_type(typeof(b), nonzero_eltype(T)), b)
    constructor(T)(map(*, ntuple(i -> bt, Val(fieldcount(T))), fields(v))...)
end
@inline Base.:*(b::Union{Zero, One}, v::T) where {T <: AbstractTensor} = constructor(T)(map(*, ntuple(i -> b, Val(fieldcount(T))), fields(v))...)
@inline Base.:*(v::AbstractTensor, b::Number) = b * v
@inline Base.:/(v::T, b::Number) where {T <: AbstractTensor} = @inline  begin
    bt = convert(promote_type(typeof(b), nonzero_eltype(T)), b)
    constructor(T)(map(/, fields(v), ntuple(i -> bt, Val(fieldcount(T))))...)
end
@inline //(v::AbstractTensor, b::Number) = (One() // b) * v
@inline Base.:-(a::T) where {T <: AbstractTensor} = @inline constructor(T)(map(-, fields(a))...)
@inline zero(::Type{T}) where {T <: AbstractTensor} = @inline constructor(T)(_zero_for_tuple(fieldtypes(T)...)...)
@inline zero(::T) where {T <: AbstractTensor} = zero(T)
@inline conj(a::T) where {T <: AbstractTensor} = @inline constructor(T)(map(conj, fields(a))...)

# Definitons for mixed Types
@inline Base.:+(a::AbstractTensor, b::AbstractTensor) = Tensor(a.x + b.x, a.y + b.y, a.z + b.z)
@inline Base.:+(a::AbstractTensor...) = Tensor(+(map(_x, a)...), +(map(_y, a)...), +(map(_z, a)...))
@inline Base.:-(a::AbstractTensor, b::AbstractTensor) = Tensor(a.x - b.x, a.y - b.y, a.z - b.z)
@inline ==(a::AbstractTensor, b::AbstractTensor) = (a.x == b.x) & (a.y == b.y) & (a.z == b.z)


# We treat Vec's as scalar for broadcasting but the default definition of + and - for AbstractArray's relies
# on broadcasting to perform addition and subtraction. The method definitons below overcomes this inconsistency
@inline Base.:+(a::AbstractTensor, b::AbstractArray) = Array(a) + b
@inline Base.:+(b::AbstractArray, a::AbstractTensor) = b + Array(a)
@inline Base.:-(a::AbstractTensor, b::AbstractArray) = Array(a) - b
@inline Base.:-(b::AbstractArray, a::AbstractTensor) = b - Array(a)

@inline function _muladd(a::Number, v::AbstractTensor, u::AbstractTensor)
    at = convert(promote_type(typeof(a), nonzero_eltype(v), nonzero_eltype(u)), a)
    return Tensor(_muladd(at, v.x, u.x), _muladd(at, v.y, u.y), _muladd(at, v.z, u.z))
end
@inline _muladd(a::Union{Zero, One}, v::AbstractTensor, u::AbstractTensor) = Tensor(_muladd(a, v.x, u.x), _muladd(a, v.y, u.y), _muladd(a, v.z, u.z))

@inline _muladd(v::AbstractTensor, a::Number, u::AbstractTensor) = _muladd(a, v, u)

@inline muladd(a::Number, v::AbstractTensor, u::AbstractTensor) = _muladd(a, v, u)
@inline muladd(v::AbstractTensor, a::Number, u::AbstractTensor) = _muladd(v, a, u)

@inline dot(a::AbstractTensor, b::AbstractTensor) = _muladd(a.x, b.x, _muladd(a.y, b.y, a.z * b.z))

@inline dotadd(a::AbstractVec, b::AbstractVec, c) = _muladd(a.x, b.x, _muladd(a.y, b.y, _muladd(a.z, b.z, c)))

@inline inner(u::AbstractVec, v::AbstractVec) = dot(conj(u), v)
@inline inneradd(u::AbstractVec, v::AbstractVec, c) = dotadd(conj(u), v, c)

@inline inneradd(T1::AbstractTensor{<:Any,N}, T2::AbstractTensor{<:Any,N}, c) where {N} =
    inneradd(T1.x, T2.x, inneradd(T1.y, T2.y, inneradd(T1.z, T2.z, c)))
@inline inner(T1::AbstractTensor{<:Any,N}, T2::AbstractTensor{<:Any,N}) where {N} = inneradd(T1.x, T2.x,
                                                                                 inneradd(T1.y, T2.y, inner(T1.z,T2.z)))

@inline Base.sum(v::AbstractVec) = v.x + v.y + v.z
@inline Base.sum(v::AbstractTensor) = sum(v.x) + sum(v.y) + sum(v.z)

@inline Base.sum(op::F, v::AbstractVec) where {F <: Function} = @inline op(v.x) + op(v.y) + op(v.z)

@inline Base.sum(op::F, v::AbstractTensor) where {F <: Function} = @inline sum(op,v.x) + sum(op,v.y) + sum(op,v.z)

@inline Base.map(f::F, vecs::Vararg{AbstractTensor{<:Any, N}}) where {F <: Function, N} = @inline(Tensor(map(f, getfield.(vecs, (:x,))...), map(f, getfield.(vecs, (:y,))...), map(f, getfield.(vecs, (:z,))...)))

