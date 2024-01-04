module TensorsLite

using Zeros

export Vec, Ten, AbstractVec, dotadd, otimes, ⊗
export 𝐢, 𝐣, 𝐤
export VecArray, ZeroArray

include("type_utils.jl")

abstract type AbstractVec{T,N} <: AbstractArray{T,N} end

struct Vec{T,N,Tx,Ty,Tz} <: AbstractVec{T,N}
    x::Tx
    y::Ty
    z::Tz

   @inline function Vec(x::Number,y::Number,z::Number)
        Tx = typeof(x)
        Ty = typeof(y)
        Tz = typeof(z)
        Tf = promote_type(Tx,Ty,Tz)
        xn = _my_convert(Tf,x)
        yn = _my_convert(Tf,y)
        zn = _my_convert(Tf,z)
        Txf = typeof(xn)
        Tyf = typeof(yn)
        Tzf = typeof(zn)
        Tff = _final_type(Txf,Tyf,Tzf)
        return new{Tff,1,Txf,Tyf,Tzf}(xn, yn, zn)
    end

    @inline function Vec(x::Vec, y::Vec, z::Vec)
        _Tx = _my_eltype(x) 
        _Ty = _my_eltype(y) 
        _Tz = _my_eltype(z) 
        Tf = promote_type(_Tx,_Ty,_Tz)
        xf = Vec(_my_convert(Tf,x.x), _my_convert(Tf,x.y), _my_convert(Tf,x.z))
        yf = Vec(_my_convert(Tf,y.x), _my_convert(Tf,y.y), _my_convert(Tf,y.z))
        zf = Vec(_my_convert(Tf,z.x), _my_convert(Tf,z.y), _my_convert(Tf,z.z))
        return new{Union{eltype(xf),eltype(yf),eltype(zf)},2,typeof(xf),typeof(yf),typeof(zf)}(xf,yf,zf)
    end
end

const AbstractTen{T} = AbstractVec{T,2}
const Vec3D{T} = Vec{T,1,T,T,T}
const Vec2Dxy{T} = Vec{Union{Zero,T},1,T,T,Zero}
const Vec2Dxz{T} = Vec{Union{Zero,T},1,T,Zero,T}
const Vec2Dyz{T} = Vec{Union{Zero,T},1,Zero,T,T}
const Vec1Dx{T} = Vec{Union{Zero,T},1,T,Zero,Zero}
const Vec1Dy{T} = Vec{Union{Zero,T},1,Zero,T,Zero}
const Vec1Dz{T} = Vec{Union{Zero,T},1,Zero,Zero,T}
const Vec0D = Vec{Zero,1,Zero,Zero,Zero}

const 𝟎⃗ = Vec(Zero(),Zero(),Zero())
const 𝐢 = Vec(One(),Zero(),Zero())
const 𝐣 = Vec(Zero(),One(),Zero())
const 𝐤 = Vec(Zero(),Zero(),One())

@inline if_zero_to_zerovec(x::Zero) = 𝟎⃗
@inline if_zero_to_zerovec(x::Vec) = x

@inline function Vec(;x::Union{Vec,Number}=𝟎,y::Union{Vec,Number}=𝟎,z::Union{Vec,Number}=𝟎)
    if all_Numbers(x,y,z)
        return Vec(x,y,z)
    else
        x1 = if_zero_to_zerovec(x)
        y1 = if_zero_to_zerovec(y)
        z1 = if_zero_to_zerovec(z)
        return Vec(x1,y1,z1)
    end
end
 
Base.IndexStyle(::Type{T}) where T<:Vec{<:Any,1} = IndexLinear()
Base.IndexStyle(::Type{T}) where T<:Vec{<:Any,2} = IndexCartesian()

@inline _zero_type(::Type{<:Number}) = Zeros.Zero
@inline _zero(::Type{<:Number}) = Zeros.Zero()
@inline _zero_type(::Type{<:AbstractVec{<:Any,1}}) = Vec0D
@inline _zero(::Type{<:AbstractVec{<:Any,1}}) = 𝟎⃗
@inline _zero(::Type{<:AbstractVec{<:Any,2}}) = Vec(x=𝟎⃗,y=𝟎⃗,z=𝟎⃗)
@inline _zero_type(::Type{<:AbstractVec{<:Any,2}}) = typeof(Vec(x=𝟎⃗,y=𝟎⃗,z=𝟎⃗))

@inline _zero(x) = _zero(typeof(x))

@inline Base.zero(u::Vec) = Vec(x=zero(u.x),y=zero(u.y),z=zero(u.z))

Base.length(::Type{<:Vec{<:Any,1}}) = 3
Base.length(u::Vec{<:Any,1}) = length(typeof(u))

Base.size(::Type{<:Vec{T,1}}) where T = (3,)
Base.size(u::Vec{T,1}) where T = size(typeof(u))

Base.length(::Type{<:AbstractTen}) = 9
Base.length(u::AbstractTen) = length(typeof(u))

Base.size(::Type{<:AbstractTen}) = (3,3)
Base.size(u::AbstractTen) = size(typeof(u))

@inline _x(u::Vec) = getfield(u,1)
@inline _y(u::Vec) = getfield(u,2)
@inline _z(u::Vec) = getfield(u,3)

@inline function Base.getindex(u::AbstractVec{<:Any,1},i::Integer)
    return getfield(u,i)
end

@inline function Base.getindex(u::AbstractTen,i::Integer,j::Integer)
    return getfield(getfield(u,j),i)
end

include("vec_arithmetic.jl")
include("tensors.jl")
include("vecarray.jl")

end
