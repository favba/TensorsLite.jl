import Base: /, //, muladd, conj, ==, zero

include("muladd_definitions.jl")

# Definitions for abstract Tensors
@inline Base.:+(a::AbstractVec) = a
@inline Base.:*(b::Number, v::T) where {T <: AbstractVec} = @inline  begin
    bt = convert(promote_type(typeof(b), nonzero_eltype(T)), b)
    constructor(T)(map(*, ntuple(i -> bt, Val(fieldcount(T))), fields(v))...)
end
@inline Base.:*(b::Union{Zero, One}, v::T) where {T <: AbstractVec} = constructor(T)(map(*, ntuple(i -> b, Val(fieldcount(T))), fields(v))...)
@inline Base.:*(v::AbstractVec, b::Number) = b * v
@inline Base.:/(v::T, b::Number) where {T <: AbstractVec} = @inline  begin
    bt = convert(promote_type(typeof(b), nonzero_eltype(T)), b)
    constructor(T)(map(/, fields(v), ntuple(i -> bt, Val(fieldcount(T))))...)
end
@inline //(v::AbstractVec, b::Number) = (One() // b) * v
@inline Base.:-(a::T) where {T <: AbstractVec} = @inline constructor(T)(map(-, fields(a))...)
@inline zero(::Type{T}) where {T <: AbstractVec} = @inline constructor(T)(_zero_for_tuple(fieldtypes(T)...)...)
@inline zero(::T) where {T <: AbstractVec} = zero(T)
@inline conj(a::T) where {T <: AbstractVec} = @inline constructor(T)(map(conj, fields(a))...)

# Definitons for mixed Types
@inline Base.:+(a::AbstractVec, b::AbstractVec) = Vec(a.x + b.x, a.y + b.y, a.z + b.z)
@inline Base.:+(a::AbstractVec...) = Vec(+(map(_x, a)...), +(map(_y, a)...), +(map(_z, a)...))
@inline Base.:-(a::AbstractVec, b::AbstractVec) = Vec(a.x - b.x, a.y - b.y, a.z - b.z)
@inline ==(a::AbstractVec, b::AbstractVec) = (a.x == b.x) & (a.y == b.y) & (a.z == b.z)


# We treat Vec's as scalar for broadcasting but the default definition of + and - for AbstractArray's relies
# on broadcasting to perform addition and subtraction. The method definitons below overcomes this inconsistency
@inline Base.:+(a::AbstractVec, b::AbstractArray) = Array(a) + b
@inline Base.:+(b::AbstractArray, a::AbstractVec) = b + Array(a)
@inline Base.:-(a::AbstractVec, b::AbstractArray) = Array(a) - b
@inline Base.:-(b::AbstractArray, a::AbstractVec) = b - Array(a)

@inline function _muladd(a::Number, v::AbstractVec, u::AbstractVec)
    at = convert(promote_type(typeof(a), nonzero_eltype(v), nonzero_eltype(u)), a)
    return Vec(_muladd(at, v.x, u.x), _muladd(at, v.y, u.y), _muladd(at, v.z, u.z))
end
@inline _muladd(a::Union{Zero, One}, v::AbstractVec, u::AbstractVec) = Vec(_muladd(a, v.x, u.x), _muladd(a, v.y, u.y), _muladd(a, v.z, u.z))

@inline _muladd(v::AbstractVec, a::Number, u::AbstractVec) = _muladd(a, v, u)

@inline muladd(a::Number, v::AbstractVec, u::AbstractVec) = _muladd(a, v, u)
@inline muladd(v::AbstractVec, a::Number, u::AbstractVec) = _muladd(v, a, u)

@inline dot(a::AbstractVec, b::AbstractVec) = _muladd(a.x, b.x, _muladd(a.y, b.y, a.z * b.z))

@inline dotadd(a::AbstractVec{<:Any, 1}, b::AbstractVec{<:Any, 1}, c::Number) = _muladd(a.x, b.x, _muladd(a.y, b.y, _muladd(a.z, b.z, c)))

@inline inner(u::Vec{T, 1}, v::Vec{T2, 1}) where {T, T2} = dot(conj(u), v)

@inline Base.sum(v::AbstractVec{<:Any, 1}) = v.x + v.y + v.z
@inline Base.sum(op::F, v::AbstractVec{<:Any, 1}) where {F <: Function} = @inline op(v.x) + op(v.y) + op(v.z)

@inline Base.map(f::F, vecs::Vararg{AbstractVec{<:Any, N}}) where {F <: Function, N} = @inline(Vec(map(f, getfield.(vecs, (:x,))...), map(f, getfield.(vecs, (:y,))...), map(f, getfield.(vecs, (:z,))...)))

