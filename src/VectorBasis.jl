module VectorBasis

using Zeros
using MuladdMacro

export Vec
export 𝐢, 𝐣, 𝐤

abstract type AbstractVec{T,N} <: AbstractArray{T,N} end

struct Vec{T,N,Tx,Ty,Tz} <: AbstractVec{T,N}
    x::Tx
    y::Ty
    z::Tz
    @inline function Vec(;x::Number=𝟎,y::Number=𝟎,z::Number=𝟎)
        Tx = typeof(x)
        Ty = typeof(y)
        Tz = typeof(z)
        Tf = promote_type(Tx,Ty,Tz)
        Txf = _keep_zero_and_one_type(Tx,Tf)
        Tyf = _keep_zero_and_one_type(Ty,Tf)
        Tzf = _keep_zero_and_one_type(Tz,Tf)
        Tff = _final_type(Txf,Tyf,Tzf)
        return new{Tff,1,Txf,Tyf,Tzf}(Txf(x),Tyf(y),Tzf(z))
    end
end

@inline _keep_zero_and_one_type(T1,Tf) = T1 === Zero ? Zero : T1 === One ? One : Tf
@inline function _final_type(Txf, Tyf, Tzf) 
    count(i->===(i,Zero),(Txf,Tyf,Tzf)) == 2 && count(i->===(i,One),(Txf,Tyf,Tzf)) == 1 && return Union{Zero,One}
    Tff = promote_type(Txf,Tyf,Tzf)
    (Txf !== Zero && Tyf !== Zero && Tzf !== Zero) ? Tff : Union{Zero,Tff}
end 

const 𝟎⃗ = Vec()
const 𝐢 = Vec(x=One())
const 𝐣 = Vec(y=One())
const 𝐤 = Vec(z=One())

@inline _zero_type(::Type{<:Number}) = Zeros.Zero
@inline _zero(::Type{<:Number}) = Zeros.Zero()
@inline _zero_type(::Type{<:Vec{<:Any,1}}) = typeof(𝟎⃗)
@inline _zero(::Type{<:Vec{<:Any,1}}) = 𝟎⃗

@inline Base.zero(u::Vec) = Vec(x=zero(u.x),y=zero(u.y),z=zero(u.z))

Base.length(::Type{<:Vec{<:Any,1}}) = 3
Base.length(u::Vec{<:Any,1}) = length(typeof(u))

Base.size(::Type{<:Vec{T,1}}) where T = (3,)
Base.size(u::Vec{T,1}) where T = size(typeof(u))

Base.length(::Type{<:Vec{<:Any,2}}) = 9
Base.length(u::Vec{<:Any,2}) = length(typeof(u))

Base.size(::Type{<:Vec{2}}) = (3,3)
Base.size(u::Vec{2}) = size(typeof(u))

@inline _x(u::Vec) = getfield(u,1)
@inline _y(u::Vec) = getfield(u,2)
@inline _z(u::Vec) = getfield(u,3)

@inline function Base.getindex(u::Vec{T,1},i::Integer) where T
    @boundscheck checkbounds(u,i)
    return getfield(u,i)
end

include("vec_arithmetic.jl")

end
