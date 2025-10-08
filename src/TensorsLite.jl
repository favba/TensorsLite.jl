module TensorsLite

using Zeros

export Zero, One
export Tensor, AbstractTensor, AbstractVec, AbstractTen

export tensor_type_3D
export tensor_type_2Dxy, tensor_type_2Dxz, tensor_type_2Dyz
export tensor_type_1Dx, tensor_type_1Dy, tensor_type_1Dz

export Vec, Vec3D, Vec2Dxy, Vec2Dxz, Vec2Dyz, Vec2D, Vec1Dx, Vec1Dy, Vec1Dz, Vec1D, VecND
export Ten, Ten3D, Ten2Dxy, Ten2Dxz, Ten2Dyz, Ten2D, Ten1Dx, Ten1Dy, Ten1Dz, Ten1D, TenND
export DiagTen3D, DiagTen2Dxy, DiagTen2Dxz, DiagTen2Dyz
export dotadd, inner, inneradd, otimes, ⊗
export 𝐢, 𝐣, 𝐤, 𝐈
export SymTen
export SymTen3D, SymTen2Dxy, SymTen2Dxz, SymTen2Dyz, SymTen1Dx, SymTen1Dy, SymTen1Dz
export AntiSymTen
export AntiSymTen3D, AntiSymTen2Dxy, AntiSymTen2Dxz, AntiSymTen2Dyz
export TensorArray, VecArray, TenArray, SymTenArray, AntiSymTenArray
#export Tensor3DArray, Tensor2DxyArray, Tensor2DxzArray, Tensor2DyzArray, Tensor1DxArray, Tensor1DyArray, Tensor1DzArray
export Vec3DArray, Vec2DxyArray, Vec2DxzArray, Vec2DyzArray, Vec1DxArray, Vec1DyArray, Vec1DzArray
export Ten3DArray, Ten2DxyArray, Ten2DxzArray, Ten2DyzArray, Ten1DxArray, Ten1DyArray, Ten1DzArray
export SymTen3DArray, SymTen2DxyArray, SymTen2DxzArray, SymTen2DyzArray, SymTen1DxArray, SymTen1DyArray, SymTen1DzArray
export AntiSymTen3DArray, AntiSymTen2DxyArray, AntiSymTen2DxzArray, AntiSymTen2DyzArray
export nonzero_eltype

# define my own *, +, - so I can extend those operators without commiting type piracy (For SIMDExt.jl)
# I'm also using my own `dot` function, and LinearAlgebra.dot is overloaded in ext/LinearAlgebraExt.jl
@inline *(a, b) = Base.:*(a, b)
@inline +(a, b) = Base.:+(a, b)
@inline -(a, b) = Base.:-(a, b)
@inline +(a) = Base.:+(a)
@inline +(a::Vararg) = Base.:+(a...)
@inline -(a) = Base.:-(a)

include("type_utils.jl")

abstract type AbstractTensor{T, N} <: AbstractArray{T, N} end

# Treat Vec's as scalar when broadcasting
Base.Broadcast.broadcastable(u::AbstractTensor) = (u,)

struct Tensor{T, N, Tx, Ty, Tz} <: AbstractTensor{T, N}
    x::Tx
    y::Ty
    z::Tz

    @inline function Tensor(x, y, z)
        any(map(x->isa(x,AbstractTensor),(x,y,z))) && throw(DimensionMismatch())
        Tx = typeof(x)
        Ty = typeof(y)
        Tz = typeof(z)
        Tf = promote_type_ignoring_Zero_and_One(Tx, Ty, Tz)
        xn = _my_convert(Tf, x)
        yn = _my_convert(Tf, y)
        zn = _my_convert(Tf, z)
        Txf = typeof(xn)
        Tyf = typeof(yn)
        Tzf = typeof(zn)
        Tff = Union{Txf,Tyf,Tzf}
        return new{Tff, 1, Txf, Tyf, Tzf}(xn, yn, zn)
    end

    @inline function Tensor(x::AbstractTensor{Tx,N}, y::AbstractTensor{Ty,N}, z::AbstractTensor{Tz,N}) where {Tx, Ty, Tz, N}
        Tf = promote_type_ignoring_Zero_and_One(_non_StaticBool_type(Tx), _non_StaticBool_type(Ty), _non_StaticBool_type(Tz))
        xf = Tensor(_eltype_convert(Tf, x.x), _eltype_convert(Tf, x.y), _eltype_convert(Tf, x.z))
        yf = Tensor(_eltype_convert(Tf, y.x), _eltype_convert(Tf, y.y), _eltype_convert(Tf, y.z))
        zf = Tensor(_eltype_convert(Tf, z.x), _eltype_convert(Tf, z.y), _eltype_convert(Tf, z.z))
        return new{Union{eltype(xf), eltype(yf), eltype(zf)}, N+1, typeof(xf), typeof(yf), typeof(zf)}(xf, yf, zf)
    end
end

########################## aliases ###########################

const AbstractVec{T} = AbstractTensor{T, 1}
const AbstractTen{T} = AbstractTensor{T, 2}
#const Vec{T,Tx,Ty,Tz} = Tensor{T,1,Tx,Ty,Tz}
const Vec3D{T} = Tensor{T,1,T,T,T}

const Vec2Dxy{T} = Tensor{Union{Zero, T}, 1, T, T, Zero}
const Vec2Dxz{T} = Tensor{Union{Zero, T}, 1, T, Zero, T}
const Vec2Dyz{T} = Tensor{Union{Zero, T}, 1, Zero, T, T}
const Vec2D{T} = Tensor{Vec2Dxy{T}, Vec2Dxz{T}, Vec2Dyz{T}}
const Vec1Dx{T} = Tensor{Union{Zero, T}, 1, T, Zero, Zero}
const Vec1Dy{T} = Tensor{Union{Zero, T}, 1, Zero, T, Zero}
const Vec1Dz{T} = Tensor{Union{Zero, T}, 1, Zero, Zero, T}
const Vec1D{T} = Union{Vec1Dx{T}, Vec1Dy{T}, Vec1Dz{T}}
const VecND{T} = Union{Vec3D{T}, Vec2D{T}, Vec1D{T}}
const Vec0D = Vec3D{Zero}

# This ends up also being a VecMaybe1Dz{T, Tz}, if T===Zero and Tz != Zero
const VecMaybe2Dxy{T, Tz} = Tensor{Union{T, Tz},1, T, T, Tz}

#const Ten{Tf,Tvx,Tvy,Tvz} = Tensor{Tf,2,Tvx,Tvy, Tvz}
const Ten3D{T} = Tensor{T,2,Vec3D{T},Vec3D{T},Vec3D{T}}
const Ten2Dxy{T} = Tensor{Union{Zero, T}, 2, Vec2Dxy{T}, Vec2Dxy{T}, Vec0D}
const Ten2Dxz{T} = Tensor{Union{Zero, T}, 2, Vec2Dxz{T}, Vec0D, Vec2Dxz{T}}
const Ten2Dyz{T} = Tensor{Union{Zero, T}, 2, Vec0D, Vec2Dyz{T}, Vec2Dyz{T}}
const Ten2D{T} = Union{Ten2Dxy{T}, Ten2Dxz{T}, Ten2Dyz{T}}
const Ten1Dx{T} = Tensor{Union{Zero, T}, 2, Vec1Dx{T}, Vec0D, Vec0D}
const Ten1Dy{T} = Tensor{Union{Zero, T}, 2, Vec0D, Vec1Dy{T}, Vec0D}
const Ten1Dz{T} = Tensor{Union{Zero, T}, 2, Vec0D, Vec0D, Vec1Dz{T}}
const Ten1D{T} = Union{Ten1Dx{T}, Ten1Dy{T}, Ten1Dz{T}}
const TenND{T} = Union{Ten3D{T}, Ten2D{T}, Ten1D{T}}
const Ten0D = Ten3D{Zero}
const DiagTen3D{T} = Tensor{Union{Zero, T}, 2, Vec1Dx{T}, Vec1Dy{T}, Vec1Dz{T}}
const DiagTen2Dxy{T} = Tensor{Union{Zero, T}, 2, Vec1Dx{T}, Vec1Dy{T}, Vec0D}
const DiagTen2Dxz{T} = Tensor{Union{Zero, T}, 2, Vec1Dx{T}, Vec0D, Vec1Dz{T}}
const DiagTen2Dyz{T} = Tensor{Union{Zero, T}, 2, Vec0D, Vec1Dy{T}, Vec1Dz{T}}

const TenMaybe2Dxy{T, Tz} = Tensor{Union{T, Tz}, 2, VecMaybe2Dxy{T, Tz}, VecMaybe2Dxy{T, Tz}, Vec3D{Tz}}

const Tensor3D{Tf,N,T} = Tensor{Tf,N,T,T,T}
## These also ends up being a Tensor1Dn if T===Zero and Tz != Zero
const Tensor2Dxy{Tf,N,T,Tz} = Tensor{Tf,N,T,T,Tz}
const Tensor2Dxz{Tf,N,T,Tz} = Tensor{Tf,N,T,Tz,T}
const Tensor2Dyz{Tf,N,T,Tz} = Tensor{Tf,N,Tz,T,T}


########################## aliases ###########################


########################## constructors ###########################

include("vec_type_utils.jl")

@inline constructor(::Type{T}) where {T <: Tensor} = Tensor

@inline Tensor{1}() = Tensor(Zero(),Zero(),Zero())
@inline Tensor{N}() where N = Tensor(Tensor{N-1}(), Tensor{N-1}(), Tensor{N-1}())

@inline check_args_ignoring_zeros(::Any,::Any,::Any) = nothing
@inline function check_args_ignoring_zeros(::Union{<:AbstractTensor{<:Any,N},Zero},::Union{<:AbstractTensor{<:Any,N},Zero},::Union{<:AbstractTensor{<:Any,N},Zero}) where {N}
    return nothing
end

@inline function check_args_ignoring_zeros(::Union{<:AbstractTensor{<:Any,N1},Zero},::Union{<:AbstractTensor{<:Any,N2},Zero},::Union{<:AbstractTensor{<:Any,N3},Zero}) where {N1,N2,N3}
    return throw(DimensionMismatch())
end

@inline _get_ndims(::Any,::Any,::Any) = Val{0}()
@inline function _get_ndims(::Union{<:AbstractTensor{<:Any,N},Zero},::Union{<:AbstractTensor{<:Any,N},Zero},::Union{<:AbstractTensor{<:Any,N},Zero}) where {N}
    return Val{N}()
end

@inline if_zero_to_tensor(::Val{0}, ::Zero) = Zero()
@inline if_zero_to_tensor(::Val{N}, ::Zero) where {N} = Tensor{N}()
@inline if_zero_to_tensor(::Val{N}, x) where {N} = x
@inline if_zero_to_tensor(v::Val{N},x,y,z) where {N} = (if_zero_to_tensor(v,x), if_zero_to_tensor(v,y), if_zero_to_tensor(v,z))

@inline function Tensor(;x=𝟎,y=𝟎,z=𝟎)
    if (x === 𝟎) && (y == 𝟎) && (z == 𝟎)
        return Tensor{1}()
    else
        check_args_ignoring_zeros(x,y,z)
        NV = _get_ndims(x,y,z)
        xf, yf, zf = if_zero_to_tensor(NV,x,y,z)
        return Tensor(xf,yf,zf)
    end
end

Vec(x,y,z) = Tensor(x,y,z)
Vec(;x=𝟎,y=𝟎,z=𝟎) = Vec(x,y,z)

Vec3D{T}(a, b, c) where {T} = Tensor(convert(T, a), convert(T, b), convert(T, c))

Vec2Dxy{T}(a, b) where {T} = Tensor(convert(T, a), convert(T, b), Zero())
Vec2Dxy(a, b) = Vec2Dxy{promote_type(typeof(a),typeof(b))}(a, b)

Vec2Dxz{T}(a, b) where {T} = Tensor(convert(T, a), Zero(), convert(T, b))
Vec2Dxz(a, b) = Vec2Dxz{promote_type(typeof(a),typeof(b))}(a, b)

Vec2Dyz{T}(a, b) where {T} = Tensor(Zero(), convert(T, a), convert(T, b))
Vec2Dyz(a, b) = Vec2Dyz{promote_type(typeof(a),typeof(b))}(a, b)

Vec1Dx{T}(a) where {T} = Tensor(convert(T, a), Zero(), Zero())
Vec1Dx(a) = Vec1Dx{typeof(a)}(a)

Vec1Dy{T}(a) where {T} = Tensor(Zero(), convert(T, a), Zero())
Vec1Dy(a) = Vec1Dy{typeof(a)}(a)

Vec1Dz{T}(a) where {T} = Tensor(Zero(), Zero(), convert(T, a))
Vec1Dz(a) = Vec1Dz{typeof(a)}(a)


############################### AbstractArray interface #############################

tensor_ndims(::Type{T}) where {T} = Val{0}()
tensor_ndims(::Type{TV}) where {T,N,TV<:Tensor{T,N}} = Val{N}()

@inline Base.convert(::Type{Tensor{T, N, Tx, Ty, Tz}}, u::AbstractTensor{T2, N}) where {T, N, Tx, Ty, Tz, T2} = Tensor(convert(Tx, u.x), convert(Ty, u.y), convert(Tz, u.z))

Base.IndexStyle(::Type{T}) where {T <: AbstractTensor{<:Any, 1}} = IndexLinear()
Base.IndexStyle(::Type{T}) where {T <: AbstractTensor{<:Any, N}} where {N} = IndexCartesian()

Base.length(::Type{<:AbstractTensor{<:Any, N}}) where {N} = 3^N
Base.length(u::AbstractTensor{<:Any, N}) where {N} = length(typeof(u))

Base.size(::Type{<:AbstractTensor{T, N}}) where {T,N} = ntuple(i -> 3, Val{N}())
Base.size(u::AbstractTensor{T, N}) where {T, N} = size(typeof(u))

@inline _x(u::AbstractTensor) = u.x
@inline _y(u::AbstractTensor) = u.y
@inline _z(u::AbstractTensor) = u.z

Base.@constprop :aggressive function Base.getindex(u::Tensor{<:Any, 1}, i::Integer)
    return getfield(u, i)
end

Base.@constprop :aggressive function Base.getindex(u::Tensor{<:Any, N}, I::Vararg{Integer, N}) where {N}
    return getindex(getfield(u, @inbounds(I[N])), ntuple(i -> @inbounds(I[i]), Val{N-1}())...)
end

Base.rand(::Type{Zero}) = Zero()
Base.rand(::Type{One}) = One()
Base.rand(::Type{Tensor{T,N,Tx,Ty,Tz}}) where {T,N,Tx,Ty,Tz} = Tensor(rand(Tx), rand(Ty), rand(Tz))

# useful compile time constant tensors
const 𝐢 = Vec(One(), Zero(), Zero())
const 𝐣 = Vec(Zero(), One(), Zero())
const 𝐤 = Vec(Zero(), Zero(), One())
const 𝐈 = Tensor(Vec1Dx(One()), Vec1Dy(One()), Vec1Dz(One()))


include("vec_arithmetic.jl")
include("tensors.jl")
include("symmetric_tensors.jl")
include("antisym_tensors.jl")
include("vecarray.jl")
include("sym_antisym_vecarray.jl")

end
