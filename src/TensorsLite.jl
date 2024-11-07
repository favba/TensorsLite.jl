module TensorsLite

using Zeros

export Vec, Ten, AbstractVec
export Vec3D, Vec2Dxy, Vec2Dxz, Vec2Dyz, Vec2D, Vec1Dx, Vec1Dy, Vec1Dz, Vec1D, VecND
export Ten3D, Ten2Dxy, Ten2Dxz, Ten2Dyz, Ten2D, Ten1Dx, Ten1Dy, Ten1Dz, Ten1D, TenND
export dotadd, inner, otimes, ⊗
export 𝐢, 𝐣, 𝐤
export SymTen
export SymTen3D, SymTen2Dxy, SymTen2Dxz, SymTen2Dyz, SymTen1Dx, SymTen1Dy, SymTen1Dz
export AntiSymTen
export AntiSymTen3D, AntiSymTen2Dxy, AntiSymTen2Dxz, AntiSymTen2Dyz
export VecArray, TenArray, SymTenArray, AntiSymTenArray
export Vec3DArray, Vec2DxyArray, Vec2DxzArray, Vec2DyzArray, Vec1DxArray, Vec1DyArray, Vec1DzArray
export Ten3DArray, Ten2DxyArray, Ten2DxzArray, Ten2DyzArray, Ten1DxArray, Ten1DyArray, Ten1DzArray
export SymTen3DArray, SymTen2DxyArray, SymTen2DxzArray, SymTen2DyzArray, SymTen1DxArray, SymTen1DyArray, SymTen1DzArray
export AntiSymTen3DArray, AntiSymTen2DxyArray, AntiSymTen2DxzArray, AntiSymTen2DyzArray
export nonzero_eltype

# define my own *, +, - so I can extend those operators without commiting type piracy (For SIMDExt.jl)
@inline *(a, b) = Base.:*(a, b)
@inline +(a, b) = Base.:+(a, b)
@inline -(a, b) = Base.:-(a, b)
@inline +(a) = Base.:+(a)
@inline +(a::Vararg) = Base.:+(a...)
@inline -(a) = Base.:-(a)

include("type_utils.jl")

abstract type AbstractVec{T, N} <: AbstractArray{T, N} end

# Treat Vec's as scalar when broadcasting
Base.Broadcast.broadcastable(u::AbstractVec) = (u,)

struct Vec{T, N, Tx, Ty, Tz} <: AbstractVec{T, N}
    x::Tx
    y::Ty
    z::Tz

    @inline function Vec(x, y, z)
        Tx = typeof(x)
        Ty = typeof(y)
        Tz = typeof(z)
        Tf = promote_type_ignoring_Zero(Tx, Ty, Tz)
        xn = _my_convert(Tf, x)
        yn = _my_convert(Tf, y)
        zn = _my_convert(Tf, z)
        Txf = typeof(xn)
        Tyf = typeof(yn)
        Tzf = typeof(zn)
        Tff = _final_type(Txf, Tyf, Tzf)
        return new{Tff, 1, Txf, Tyf, Tzf}(xn, yn, zn)
    end

    @inline function Vec(x::Vec, y::Vec, z::Vec)
        _Tx = nonzero_eltype(x)
        _Ty = nonzero_eltype(y)
        _Tz = nonzero_eltype(z)
        Tf = promote_type_ignoring_Zero(_Tx, _Ty, _Tz)
        xf = Vec(_my_convert(Tf, x.x), _my_convert(Tf, x.y), _my_convert(Tf, x.z))
        yf = Vec(_my_convert(Tf, y.x), _my_convert(Tf, y.y), _my_convert(Tf, y.z))
        zf = Vec(_my_convert(Tf, z.x), _my_convert(Tf, z.y), _my_convert(Tf, z.z))
        return new{Union{eltype(xf), eltype(yf), eltype(zf)}, 2, typeof(xf), typeof(yf), typeof(zf)}(xf, yf, zf)
    end
end

include("vec_type_utils.jl")

@inline Base.convert(::Type{Vec{T, N, Tx, Ty, Tz}}, u::AbstractVec{T2, N}) where {T, N, Tx, Ty, Tz, T2} = Vec(convert(Tx, u.x), convert(Ty, u.y), convert(Tz, u.z))

@inline constructor(::Type{T}) where {T <: Vec} = Vec

const AbstractTen{T} = AbstractVec{T, 2}

const Vec3D{T} = Vec{T, 1, T, T, T}
const Vec2Dxy{T} = Vec{Union{Zero, T}, 1, T, T, Zero}
const Vec2Dxz{T} = Vec{Union{Zero, T}, 1, T, Zero, T}
const Vec2Dyz{T} = Vec{Union{Zero, T}, 1, Zero, T, T}
const Vec2D{T} = Union{Vec2Dxy{T}, Vec2Dxz{T}, Vec2Dyz{T}}
const Vec1Dx{T} = Vec{Union{Zero, T}, 1, T, Zero, Zero}
const Vec1Dy{T} = Vec{Union{Zero, T}, 1, Zero, T, Zero}
const Vec1Dz{T} = Vec{Union{Zero, T}, 1, Zero, Zero, T}
const Vec1D{T} = Union{Vec1Dx{T}, Vec1Dy{T}, Vec1Dz{T}}
const VecND{T} = Union{Vec3D{T}, Vec2D{T}, Vec1D{T}}
const Vec0D = VecND{Zero}

# This ends up also being a VecMaybe1Dz{T, Tz}, if T===Zero and Tz != Zero
const VecMaybe2Dxy{T, Tz} = Vec{Union{T, Tz}, 1, T, T, Tz}

const Ten3D{T} = Vec{T, 2, Vec3D{T}, Vec3D{T}, Vec3D{T}}
const Ten2Dxy{T} = Vec{Union{Zero, T}, 2, Vec2Dxy{T}, Vec2Dxy{T}, Vec0D}
const Ten2Dxz{T} = Vec{Union{Zero, T}, 2, Vec2Dxz{T}, Vec0D, Vec2Dxz{T}}
const Ten2Dyz{T} = Vec{Union{Zero, T}, 2, Vec0D, Vec2Dyz{T}, Vec2Dyz{T}}
const Ten2D{T} = Union{Ten2Dxy{T}, Ten2Dxz{T}, Ten2Dyz{T}}
const Ten1Dx{T} = Vec{Union{Zero, T}, 2, Vec1Dx{T}, Vec0D, Vec0D}
const Ten1Dy{T} = Vec{Union{Zero, T}, 2, Vec0D, Vec1Dy{T}, Vec0D}
const Ten1Dz{T} = Vec{Union{Zero, T}, 2, Vec0D, Vec0D, Vec1Dz{T}}
const Ten1D{T} = Union{Ten1Dx{T}, Ten1Dy{T}, Ten1Dz{T}}
const TenND{T} = Union{Ten3D{T}, Ten2D{T}, Ten1D{T}}
const Ten0D = TenND{Zero}

const TenMaybe2Dxy{T, Tz} = Vec{Union{T, Tz}, 2, VecMaybe2Dxy{T, Tz}, VecMaybe2Dxy{T, Tz}, Vec3D{Tz}}

const 𝐢 = Vec(One(), Zero(), Zero())
const 𝐣 = Vec(Zero(), One(), Zero())
const 𝐤 = Vec(Zero(), Zero(), One())

@inline if_zero_to_zerovec(::Zero) = Vec(𝟎, 𝟎, 𝟎)
@inline if_zero_to_zerovec(x::Vec) = x

@inline function Vec(; x = 𝟎, y = 𝟎, z = 𝟎)
    if no_Vecs(x, y, z)
        return Vec(x, y, z)
    else
        x1 = if_zero_to_zerovec(x)
        y1 = if_zero_to_zerovec(y)
        z1 = if_zero_to_zerovec(z)
        return Vec(x1, y1, z1)
    end
end

Base.IndexStyle(::Type{T}) where {T <: Vec{<:Any, 1}} = IndexLinear()
Base.IndexStyle(::Type{T}) where {T <: Vec{<:Any, 2}} = IndexCartesian()

Base.length(::Type{<:Vec{<:Any, 1}}) = 3
Base.length(u::Vec{<:Any, 1}) = length(typeof(u))

Base.size(::Type{<:Vec{T, 1}}) where {T} = (3,)
Base.size(u::Vec{T, 1}) where {T} = size(typeof(u))

Base.length(::Type{<:AbstractTen}) = 9
Base.length(u::AbstractTen) = length(typeof(u))

Base.size(::Type{<:AbstractTen}) = (3, 3)
Base.size(u::AbstractTen) = size(typeof(u))

@inline _x(u::AbstractVec) = u.x
@inline _y(u::AbstractVec) = u.y
@inline _z(u::AbstractVec) = u.z

@inline function Base.getindex(u::Vec{<:Any, 1}, i::Integer)
    return getfield(u, i)
end

@inline function Base.getindex(u::Vec{<:Any, 2}, i::Integer, j::Integer)
    return getfield(getfield(u, j), i)
end

include("vec_arithmetic.jl")
include("tensors.jl")
include("symmetric_tensors.jl")
include("antisym_tensors.jl")
include("vecarray.jl")
include("sym_antisym_vecarray.jl")

end
