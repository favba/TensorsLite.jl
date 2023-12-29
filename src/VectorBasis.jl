module VectorBasis

using Zeros
using MuladdMacro

export Vec, Ten, AbstractVec, dotadd, otimes, ⊗
export 𝐢, 𝐣, 𝐤

abstract type AbstractVec{T,N} <: AbstractArray{T,N} end

@inline _keep_zero_and_one_type(T1,Tf) = T1 === Zero ? Zero : T1 === One ? One : Tf
@inline function _final_type(Txf, Tyf, Tzf) 
    count(i->===(i,Zero),(Txf,Tyf,Tzf)) == 2 && count(i->===(i,One),(Txf,Tyf,Tzf)) == 1 && return Union{Zero,One}
    Tff = promote_type(Txf,Tyf,Tzf)
    (Txf !== Zero && Tyf !== Zero && Tzf !== Zero) ? Tff : Union{Zero,Tff}
end 

struct Vec{T,N,Tx,Ty,Tz} <: AbstractVec{T,N}
    x::Tx
    y::Ty
    z::Tz

   @inline function Vec(x::Number,y::Number,z::Number)
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

    @inline function Vec(x::Vec, y::Vec, z::Vec)
        _Tx = _my_eltype(x) 
        _Ty = _my_eltype(y) 
        _Tz = _my_eltype(z) 
        Tf = promote_type(_Tx,_Ty,_Tz)
        xf = Vec(_my_convert(Tf,x.x), _my_convert(Tf,x.y), _my_convert(Tf,x.z))
        yf = Vec(_my_convert(Tf,y.x), _my_convert(Tf,y.y), _my_convert(Tf,y.z))
        zf = Vec(_my_convert(Tf,z.x), _my_convert(Tf,z.y), _my_convert(Tf,z.z))
        return new{eltype(xf),2,typeof(xf),typeof(yf),typeof(zf)}(xf,yf,zf)
    end
end

const AbstractTen{T} = AbstractVec{T,2}

const 𝟎⃗ = Vec(Zero(),Zero(),Zero())
const 𝐢 = Vec(One(),Zero(),Zero())
const 𝐣 = Vec(Zero(),One(),Zero())
const 𝐤 = Vec(Zero(),Zero(),One())

@inline all_Numbers(a::T1,y::T2,z::T3) where {T1,T2,T3} = T1 <: Number && T2 <: Number && T3 <: Number

@inline zero_to_zerovec(x::Zero) = 𝟎⃗
@inline zero_to_zerovec(x::Vec) = x

@inline function Vec(;x::Union{Vec,Number}=𝟎,y::Union{Vec,Number}=𝟎,z::Union{Vec,Number}=𝟎)
    if all_Numbers(x,y,z)
        return Vec(x,y,z)
    else
        x1 = zero_to_zerovec(x)
        y1 = zero_to_zerovec(y)
        z1 = zero_to_zerovec(z)
        return Vec(x1,y1,z1)
    end
end
 
Base.IndexStyle(::Type{T}) where T<:Vec{<:Any,1} = IndexLinear()
Base.IndexStyle(::Type{T}) where T<:Vec{<:Any,2} = IndexCartesian()

function _my_convert(T::Type,x::T1) where T1
    T1 === Zero && return x
    T1 === One && return x
    return convert(T,x)
end

@inline function _non_zero_type(T::Type)
    if T isa Union
        T.a !== Zero && return T.a
        T.b !== Zero && return T.b
    else
        return T
    end
end

@inline function _my_eltype(x::Vec)
    T = eltype(x)
    return _non_zero_type(T)
end


@inline _zero_type(::Type{<:Number}) = Zeros.Zero
@inline _zero(::Type{<:Number}) = Zeros.Zero()
@inline _zero_type(::Type{<:AbstractVec{<:Any,1}}) = typeof(𝟎⃗)
@inline _zero(::Type{<:AbstractVec{<:Any,1}}) = 𝟎⃗
@inline _zero(::Type{<:AbstractVec{<:Any,2}}) = Vec(x=𝟎⃗,y=𝟎⃗,z=𝟎⃗)
@inline _zero_type(::Type{<:AbstractVec{<:Any,2}}) = typeof(Vec(x=𝟎⃗,y=𝟎⃗,z=𝟎⃗))

@inline _zero(x) = _zero(typeof(x))

@inline Base.zero(u::Vec) = Vec(x=zero(u.x),y=zero(u.y),z=zero(u.z))

Base.length(::Type{<:Vec{<:Any,1}}) = 3
Base.length(u::Vec{<:Any,1}) = length(typeof(u))

Base.size(::Type{<:Vec{T,1}}) where T = (3,)
Base.size(u::Vec{T,1}) where T = size(typeof(u))

Base.length(::Type{<:Vec{<:Any,2}}) = 9
Base.length(u::Vec{<:Any,2}) = length(typeof(u))

Base.size(::Type{<:Vec{<:Any,2}}) = (3,3)
Base.size(u::Vec{<:Any,2}) = size(typeof(u))

@inline _x(u::Vec) = getfield(u,1)
@inline _y(u::Vec) = getfield(u,2)
@inline _z(u::Vec) = getfield(u,3)

@inline function Base.getindex(u::Vec{T,1},i::Integer) where T
    return getfield(u,i)
end

@inline function Base.getindex(u::Vec{T,2},i::Integer,j::Integer) where T
    return getfield(getfield(u,j),i)
end

include("vec_arithmetic.jl")
include("tensors.jl")
end
