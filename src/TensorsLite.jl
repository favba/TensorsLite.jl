module TensorsLite

using Zeros

export Vec, Ten, AbstractVec, AbstractTen
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

    @inline function Vec(x::AbstractVec{Tx,N}, y::AbstractVec{Ty,N}, z::AbstractVec{Tz,N}) where {Tx, Ty, Tz, N}
        Tf = promote_type_ignoring_Zero(_non_zero_type(Tx), _non_zero_type(Ty), _non_zero_type(Tz))
        xf = Vec(_eltype_convert(Tf, x.x), _eltype_convert(Tf, x.y), _eltype_convert(Tf, x.z))
        yf = Vec(_eltype_convert(Tf, y.x), _eltype_convert(Tf, y.y), _eltype_convert(Tf, y.z))
        zf = Vec(_eltype_convert(Tf, z.x), _eltype_convert(Tf, z.y), _eltype_convert(Tf, z.z))
        return new{Union{eltype(xf), eltype(yf), eltype(zf)}, N+1, typeof(xf), typeof(yf), typeof(zf)}(xf, yf, zf)
    end
end

@inline Vec() = Vec(Zero(), Zero(), Zero())
@inline Vec{1}() = Vec()
@inline Vec{N}() where N = Vec(Vec{N-1}(), Vec{N-1}(), Vec{N-1}())

include("vec_type_utils.jl")

@inline Base.convert(::Type{Vec{T, N, Tx, Ty, Tz}}, u::AbstractVec{T2, N}) where {T, N, Tx, Ty, Tz, T2} = Vec(convert(Tx, u.x), convert(Ty, u.y), convert(Tz, u.z))

@inline constructor(::Type{T}) where {T <: Vec} = Vec

const AbstractTen{T} = AbstractVec{T, 2}

const Vec3D{T} = Vec{T,1,T,T,T}

const Vec2Dxy{T} = Vec{Union{Zero, T}, 1, T, T, Zero}
Vec2Dxy{T}(a, b) where {T} = Vec(convert(T, a), convert(T, b), Zero())
Vec2Dxy(a, b) = Vec2Dxy{promote_type(typeof(a),typeof(b))}(a, b)
Vec2Dxy{T}(a::AbstractVec{<:Any, N}, b::AbstractVec{<:Any, N}) where {T, N} = Vec(_eltype_convert(T, a), _eltype_convert(T, b), Vec{N}())
Vec2Dxy(a::AbstractVec{Ta, N}, b::AbstractVec{Tb, N}) where {Ta, Tb, N} = Vec2Dxy{promote_type(_non_zero_type(Ta), _non_zero_type(Tb))}(a, b)

const Vec2Dxz{T} = Vec{Union{Zero, T}, 1, T, Zero, T}
Vec2Dxz{T}(a, b) where {T} = Vec(convert(T, a), Zero(), convert(T, b))
Vec2Dxz(a, b) = Vec2Dxz{promote_type(typeof(a),typeof(b))}(a, b)
Vec2Dxz{T}(a::AbstractVec{<:Any, N}, b::AbstractVec{<:Any, N}) where {T, N} = Vec(_eltype_convert(T, a), Vec{N}(), _eltype_convert(T, b))
Vec2Dxz(a::AbstractVec{Ta, N}, b::AbstractVec{Tb, N}) where {Ta, Tb, N} = Vec2Dxz{promote_type(_non_zero_type(Ta), _non_zero_type(Tb))}(a, b)

const Vec2Dyz{T} = Vec{Union{Zero, T}, 1, Zero, T, T}
Vec2Dyz{T}(a, b) where {T} = Vec(Zero(), convert(T, a), convert(T, b))
Vec2Dyz(a, b) = Vec2Dyz{promote_type(typeof(a),typeof(b))}(a, b)
Vec2Dyz{T}(a::AbstractVec{<:Any, N}, b::AbstractVec{<:Any, N}) where {T, N} = Vec(Vec{N}(), _eltype_convert(T, a), _eltype_convert(T, b))
Vec2Dyz(a::AbstractVec{Ta, N}, b::AbstractVec{Tb, N}) where {Ta, Tb, N} = Vec2Dyz{promote_type(_non_zero_type(Ta), _non_zero_type(Tb))}(a, b)

const Vec2D{T} = Union{Vec2Dxy{T}, Vec2Dxz{T}, Vec2Dyz{T}}

const Vec1Dx{T} = Vec{Union{Zero, T}, 1, T, Zero, Zero}
Vec1Dx{T}(a) where {T} = Vec(convert(T, a), Zero(), Zero())
Vec1Dx(a) = Vec1Dx{typeof(a)}(a)
Vec1Dx{T}(a::AbstractVec{<:Any, N}) where {T, N} = Vec(_eltype_convert(T, a), Vec{N}(), Vec{N}())
Vec1Dx(a::AbstractVec{Ta, N}) where {Ta, N} = Vec1Dx{_non_zero_type(Ta)}(a)

const Vec1Dy{T} = Vec{Union{Zero, T}, 1, Zero, T, Zero}
Vec1Dy{T}(a) where {T} = Vec(Zero(), convert(T, a), Zero())
Vec1Dy(a) = Vec1Dy{typeof(a)}(a)
Vec1Dy{T}(a::AbstractVec{<:Any, N}) where {T, N} = Vec(Vec{N}(), _eltype_convert(T, a), Vec{N}())
Vec1Dy(a::AbstractVec{Ta, N}) where {Ta, N} = Vec1Dy{_non_zero_type(Ta)}(a)

const Vec1Dz{T} = Vec{Union{Zero, T}, 1, Zero, Zero, T}
Vec1Dz{T}(a) where {T} = Vec(Zero(), Zero(), convert(T, a))
Vec1Dz(a) = Vec1Dz{typeof(a)}(a)
Vec1Dz{T}(a::AbstractVec{<:Any, N}) where {T, N} = Vec(Vec{N}(), Vec{N}(), _eltype_convert(T, a))
Vec1Dz(a::AbstractVec{Ta, N}) where {Ta, N} = Vec1Dz{_non_zero_type(Ta)}(a)

const Vec1D{T} = Union{Vec1Dx{T}, Vec1Dy{T}, Vec1Dz{T}}
const VecND{T} = Union{Vec3D{T}, Vec2D{T}, Vec1D{T}}
const Vec0D = VecND{Zero}

# This ends up also being a VecMaybe1Dz{T, Tz}, if T===Zero and Tz != Zero
const VecMaybe2Dxy{T, Tz} = Vec{Union{T, Tz}, 1, T, T, Tz}

const 𝐢 = Vec(One(), Zero(), Zero())
const 𝐣 = Vec(Zero(), One(), Zero())
const 𝐤 = Vec(Zero(), Zero(), One())

Base.IndexStyle(::Type{T}) where {T <: AbstractVec{<:Any, 1}} = IndexLinear()
Base.IndexStyle(::Type{T}) where {T <: AbstractVec{<:Any, N}} where {N} = IndexCartesian()

Base.length(::Type{<:AbstractVec{<:Any, N}}) where {N} = 3^N
Base.length(u::AbstractVec{<:Any, N}) where {N} = length(typeof(u))

Base.size(::Type{<:AbstractVec{T, N}}) where {T,N} = ntuple(i -> 3, Val{N}())
Base.size(u::AbstractVec{T, N}) where {T, N} = size(typeof(u))

@inline _x(u::AbstractVec) = u.x
@inline _y(u::AbstractVec) = u.y
@inline _z(u::AbstractVec) = u.z

@inline function Base.getindex(u::Vec{<:Any, 1}, i::Integer)
    return getfield(u, i)
end

@inline function Base.getindex(u::Vec{<:Any, N}, I::Vararg{Integer, N}) where {N}
    return getindex(getfield(u, @inbounds(I[N])), ntuple(i -> @inbounds(I[i]), Val{N-1}())...)
end

include("vec_arithmetic.jl")
include("tensors.jl")
include("symmetric_tensors.jl")
include("antisym_tensors.jl")
include("vecarray.jl")
include("sym_antisym_vecarray.jl")

end
