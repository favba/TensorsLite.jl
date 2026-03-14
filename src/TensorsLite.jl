module TensorsLite

using Zeros

export Zero, One
export Tensor, AbstractTensor, Vec, Ten

export tensor_type_3D
export tensor_type_2Dxy, tensor_type_2Dxz, tensor_type_2Dyz
export tensor_type_1Dx, tensor_type_1Dy, tensor_type_1Dz

export Vec3D, Vec2Dxy, Vec2Dxz, Vec2Dyz, Vec2D, Vec1Dx, Vec1Dy, Vec1Dz, Vec1D, VecND
export Ten3D, Ten2Dxy, Ten2Dxz, Ten2Dyz, Ten2D, Ten1Dx, Ten1Dy, Ten1Dz, Ten1D, TenND
export DiagTen3D, DiagTen2Dxy, DiagTen2Dxz, DiagTen2Dyz
export dotadd, inner, inneradd, otimes, ⊗
export 𝐢, 𝐣, 𝐤, 𝐈
export SymmetricTensor, SymTen
export SymTen3D, SymTen2Dxy, SymTen2Dxz, SymTen2Dyz, SymTen1Dx, SymTen1Dy, SymTen1Dz
export AntiSymmetricTensor, AntiSymTen
export AntiSymTen3D, AntiSymTen2Dxy, AntiSymTen2Dxz, AntiSymTen2Dyz
export AbstractTensorArray, TensorArray, VecArray, TenArray, SymmetricTensorArray, AntiSymmetricTensorArray
export Vec3DArray, Vec2DxyArray, Vec2DxzArray, Vec2DyzArray, Vec1DxArray, Vec1DyArray, Vec1DzArray
export Ten3DArray, Ten2DxyArray, Ten2DxzArray, Ten2DyzArray, Ten1DxArray, Ten1DyArray, Ten1DzArray
export SymmetricTensorArray, SymTen3DArray, SymTen2DxyArray, SymTen2DxzArray, SymTen2DyzArray, SymTen1DxArray, SymTen1DyArray, SymTen1DzArray
export AntiSymmetricTensorArray, AntiSymTen3DArray, AntiSymTen2DxyArray, AntiSymTen2DxzArray, AntiSymTen2DyzArray
export nonzero_eltype

include("type_utils.jl")

"""
    AbstractTensor{N, T} <: AbstractArray{T, N}

Supertype of all Tensor types. Represents any `N`th order tensor with eltype `T`.
"""
abstract type AbstractTensor{N, T} <: AbstractArray{T, N} end

# Treat Vec's as scalar when broadcasting
Base.Broadcast.broadcastable(u::AbstractTensor) = (u,)

"""
    Tensor{N, T, Tx, Ty, Tz} <: AbstractTensor{N, T}

A `N`th order tensor with eltype `T`.
Higher order tensors are implemented as vectors of lower order tensors.

"""
struct Tensor{N, T, Tx, Ty, Tz} <: AbstractTensor{N, T}
    x::Tx
    y::Ty
    z::Tz

    @inline function Tensor(x, y, z)

        # AbstractTensor are only valid as input when they are all of the same order
        # Same order Tensors as input are dealt with in the next method definition
        x isa AbstractTensor && throw(DimensionMismatch())
        y isa AbstractTensor && throw(DimensionMismatch())
        z isa AbstractTensor && throw(DimensionMismatch())

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
        return new{1, Tff, Txf, Tyf, Tzf}(xn, yn, zn)
    end

    @inline function Tensor(x::AbstractTensor{N,Tx}, y::AbstractTensor{N,Ty}, z::AbstractTensor{N,Tz}) where {N, Tx, Ty, Tz}
        Tf = promote_type_ignoring_Zero_and_One(_non_StaticBool_type(Tx), _non_StaticBool_type(Ty), _non_StaticBool_type(Tz))
        xf = _eltype_convert(Tf, x)
        yf = _eltype_convert(Tf, y)
        zf = _eltype_convert(Tf, z)
        return new{N+1, Union{eltype(xf), eltype(yf), eltype(zf)}, typeof(xf), typeof(yf), typeof(zf)}(xf, yf, zf)
    end
end

@inline constructor(::Type{T}) where {T <: Tensor} = Tensor

########################## constructors ###########################

include("vec_type_utils.jl")

@inline Tensor{0}() = 𝟎
@inline Tensor{1}() = Tensor(Zero(),Zero(),Zero())
@inline Tensor{N}() where N = Tensor(Tensor{N-1}(), Tensor{N-1}(), Tensor{N-1}())

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

include("aliases.jl")

############################### AbstractArray interface #############################

@inline Base.convert(::Type{Tensor{N, T, Tx, Ty, Tz}}, u::AbstractTensor{N, T2}) where {T, N, Tx, Ty, Tz, T2} = Tensor(convert(Tx, u.x), convert(Ty, u.y), convert(Tz, u.z))

Base.IndexStyle(::Type{T}) where {T <: Vec} = IndexLinear()
Base.IndexStyle(::Type{T}) where {T <: AbstractTensor{N}} where {N} = IndexCartesian()

Base.length(::Type{<:AbstractTensor{N}}) where {N} = 3^N
Base.length(u::AbstractTensor) = length(typeof(u))

Base.size(::Type{<:AbstractTensor{N}}) where {N} = ntuple(i -> 3, Val{N}())
Base.size(u::AbstractTensor) = size(typeof(u))

@inline _x(u::AbstractTensor) = u.x
@inline _y(u::AbstractTensor) = u.y
@inline _z(u::AbstractTensor) = u.z

Base.@constprop :aggressive __checkbounds(i::Integer) = (0 < i <= 3)

Base.@constprop :aggressive __checkbounds(i::Integer, I::Vararg{Integer}) = __checkbounds(i) && __checkbounds(I...)

Base.@constprop :aggressive function _checkbounds(a::Vec, i::Integer)
    __checkbounds(i) || throw(BoundsError(a,i))
    return nothing
end

Base.@constprop :aggressive function Base.getindex(u::Tensor{1}, i::Integer)
    @boundscheck _checkbounds(u,i)
    return @inbounds(getfield(u, i))
end

Base.@constprop :aggressive function _checkbounds(a::AbstractTensor{N}, I::Vararg{Integer,N}) where {N}
    __checkbounds(I...) || throw(BoundsError(a,I))
    return nothing
end

Base.@constprop :aggressive function Base.getindex(u::Tensor{N}, I::Vararg{Integer, N}) where {N}
    @boundscheck _checkbounds(u,I...)
    return @inbounds(getindex(getfield(u, @inbounds(I[N])), ntuple(i -> @inbounds(I[i]), Val{N-1}())...))
end

@inline @generated function Base.getproperty(T::Tensor{N}, s::Symbol) where N
    if N >= 2
        return quote
            if s === :xx
                return getfield(getfield(T, :x), :x)
            elseif s === :xy
                return getfield(getfield(T, :y), :x)
            elseif s === :xz
                return getfield(getfield(T, :z), :x)
            elseif s === :yx
                return getfield(getfield(T, :x), :y)
            elseif s === :yy
                return getfield(getfield(T, :y), :y)
            elseif s === :yz
                return getfield(getfield(T, :z), :y)
            elseif s === :zx
                return getfield(getfield(T, :x), :z)
            elseif s === :zy
                return getfield(getfield(T, :y), :z)
            elseif s === :zz
                return getfield(getfield(T, :z), :z)
            else
                return getfield(T, s)
            end
        end
    else
        return :(getfield(T,s))
    end
end

Base.rand(::Type{Zero}) = Zero()
Base.rand(::Type{One}) = One()
Base.rand(::Type{Tensor{N,T,Tx,Ty,Tz}}) where {T,N,Tx,Ty,Tz} = Tensor(rand(Tx), rand(Ty), rand(Tz))

# useful compile time constant tensors
const 𝐢 = Vec(One(), Zero(), Zero())
const 𝐣 = Vec(Zero(), One(), Zero())
const 𝐤 = Vec(Zero(), Zero(), One())
const 𝐈 = Tensor(Vec1Dx(One()), Vec1Dy(One()), Vec1Dz(One()))

include("operations.jl")

include("symmetric_tensors.jl")

include("antisym_tensors.jl")

include("vecarray.jl")

include("sym_antisym_vecarray.jl")

#### some useful Base methods #######

@inline Base.sum(v::AbstractTensor) = sum(v.x) + sum(v.y) + sum(v.z)

@inline Base.sum(op::F, v::AbstractTensor) where {F <: Function} = @inline sum(op,v.x) + sum(op,v.y) + sum(op,v.z)

@inline Base.map(f::F, vecs::Vararg{AbstractTensor{N}}) where {F <: Function, N} = @inline(Tensor(map(f, _x.(vecs)...), map(f, _y.(vecs)...), map(f, _z.(vecs)...)))

#### AbstractMatrix interface #######

@inline function Base.transpose(T::Ten)
    x = T.x
    y = T.y
    z = T.z
    nx = Tensor(x.x, y.x, z.x)
    ny = Tensor(x.y, y.y, z.y)
    nz = Tensor(x.z, y.z, z.z)
    return Tensor(nx, ny, nz)
end

@inline Base.transpose(S::SymTen) = S

@inline Base.transpose(W::AntiSymTen) = -W

@inline Base.adjoint(T::Ten) = conj(transpose(T))

end
