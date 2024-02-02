import Base: +, -, *, /, //, muladd, conj, ==, zero
import LinearAlgebra: dot, ⋅, norm, cross, ×, normalize
export dot, ⋅, norm, cross, ×, normalize # Reexport from LinearAlgebra

include("muladd_definitions.jl")

# Definitions for abstract Tensors
@inline +(a::AbstractVec) = a
@inline *(b::Number, v::T) where T<:AbstractVec = @inline  begin 
    bt = promote_type(typeof(b),_my_eltype(T))(b)
    constructor(T)(map(*,ntuple(i->bt,Val(fieldcount(T))), fields(v))...)
end
@inline *(v::AbstractVec, b::Number) = b*v
@inline /(v::AbstractVec,b::Number) = inv(b)*v
@inline //(v::AbstractVec,b::Number) = (One()//b)*v
@inline -(a::T) where T<:AbstractVec = @inline constructor(T)(map(-,fields(a))...)
@inline zero(::Type{T}) where T<:AbstractVec = @inline constructor(T)(_zero_for_tuple(fieldtypes(T)...)...)
@inline zero(::T) where T<:AbstractVec = zero(T)
@inline conj(a::T) where T<:AbstractVec = @inline constructor(T)(map(conj,fields(a))...)

# Definitons for mixed Types
@inline +(a::AbstractVec, b::AbstractVec) = Vec(a.x+b.x, a.y+b.y, a.z+b.z)
@inline +(a::AbstractVec...) = Vec(+(map(_x,a)...), +(map(_y,a)...), +(map(_z,a)...))
@inline -(a::AbstractVec, b::AbstractVec) = Vec(a.x-b.x, a.y-b.y, a.z-b.z)
@inline ==(a::AbstractVec,b::AbstractVec) = (a.x == b.x) & (a.y == b.y) & (a.z == b.z)

# We treat Vec's as scalar for broadcasting but the default definition of + and - for AbstractArray's relies
# on broadcasting to perform addition and subtraction. The method definitons below overcomes this inconsistency
@inline +(a::AbstractVec, b::AbstractArray) = Array(a) + b
@inline +(b::AbstractArray, a::AbstractVec) = b + Array(a)
@inline -(a::AbstractVec, b::AbstractArray) = Array(a) - b
@inline -(b::AbstractArray, a::AbstractVec) = b - Array(a)

@inline function _muladd(a::Number, v::AbstractVec, u::AbstractVec)
    at = promote_type(typeof(a),_my_eltype(v),_my_eltype(u))(a)
    return Vec(_muladd(at,v.x,u.x), _muladd(at,v.y,u.y), _muladd(at,v.z,u.z))
end

@inline _muladd(v::AbstractVec, a::Number, u::AbstractVec) = _muladd(a,v,u)

@inline muladd(a::Number, v::AbstractVec, u::AbstractVec) = _muladd(a,v,u)
@inline muladd(v::AbstractVec, a::Number, u::AbstractVec) = _muladd(v,a,u)

@inline dot(a::AbstractVec,b::AbstractVec) = _muladd(a.x, b.x, _muladd(a.y, b.y, a.z*b.z))

@inline fsqrt(x) = @fastmath sqrt(x)

@inline inner(u::Vec{T,1},v::Vec{T2,1}) where {T,T2} = dot(conj(u),v)

@inline norm(u::AbstractVec) = fsqrt(real(inner(u,u)))

@inline norm(u::Vec1Dx,p::Real=2) = abs(u.x)
@inline norm(u::Vec1Dy,p::Real=2) = abs(u.y)
@inline norm(u::Vec1Dz,p::Real=2) = abs(u.z)
@inline norm(u::Vec0D,p::Real=2) = 𝟎

@inline normalize(u::AbstractVec) = u/norm(u)
@inline normalize(u::AbstractVec{Zero}) = u

@inline function cross(a::AbstractVec,b::AbstractVec) 
    ax = a.x
    ay = a.y
    az = a.z
    bx = b.x
    by = b.y
    bz = b.z
    return  Vec(_muladd(ay, bz, -az*by), _muladd(az,bx, -ax*bz), _muladd(ax,by, -ay*bx))
end
