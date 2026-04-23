module TensorsLite

using Zeros

# Compile-time constant zero and one from Zeros.jl
export Zero, One

#Abstract Tensor types
export AbstractTensor, AbstractSymmetricTensor, AbstractAntiSymmetricTensor

#Abstract Tensor types aliases
export Vec, Ten, SymTen, AntiSymTen

#Concrete Tensor types defined in this package
export Tensor, SymmetricTensor, AntiSymmetricTensor

#Concrete aliases of Tensor types defined in this package
export Vec3D, Vec2Dxy, Vec2Dxz, Vec2Dyz, Vec1Dx, Vec1Dy, Vec1Dz
export Ten3D, Ten2Dxy, Ten2Dxz, Ten2Dyz, Ten1Dx, Ten1Dy, Ten1Dz
export DiagTen3D, DiagTen2Dxy, DiagTen2Dxz, DiagTen2Dyz
export SymTen3D, SymTen2Dxy, SymTen2Dxz, SymTen2Dyz, SymTen1Dx, SymTen1Dy, SymTen1Dz
export DiagSymTen3D, DiagSymTen2Dxy, DiagSymTen2Dxz, DiagSymTen2Dyz
export AntiSymTen3D, AntiSymTen2Dxy, AntiSymTen2Dxz, AntiSymTen2Dyz

#Useful Union Tensor types (for method dispatch)
export Vec2D, Vec1D, VecND, Ten2D, Ten1D, TenND
export SymTen2D, SymTen1D, AntiSymTen2D
 
#Useful compile-time constant vectors and second order Tensors
export 𝐢, 𝐣, 𝐤 # Canonical vector base
export 𝐈 # Idetity Matrix
export 𝐢𝐢, 𝐢𝐣, 𝐢𝐤
export 𝐣𝐢, 𝐣𝐣, 𝐣𝐤
export 𝐤𝐢, 𝐤𝐣, 𝐤𝐤

#Tensor operators
export dotadd, inner, inneradd, otimes, ⊗, dcontract, ⊡, dcontractadd

#Abstract SOA Array of Tensor types
export AbstractTensorArray

#Abstract SOA Array of Tensor types aliases 
export VecArray, TenArray, SymTenArray, AntiSymTenArray 

#Concrete SOA Array of Tensors types
export TensorArray, SymmetricTensorArray, AntiSymmetricTensorArray

#Concrete SOA Array of Tensors type aliases
export Vec3DArray, Vec2DxyArray, Vec2DxzArray, Vec2DyzArray, Vec1DxArray, Vec1DyArray, Vec1DzArray
export Ten3DArray, Ten2DxyArray, Ten2DxzArray, Ten2DyzArray, Ten1DxArray, Ten1DyArray, Ten1DzArray
export SymTen3DArray, SymTen2DxyArray, SymTen2DxzArray, SymTen2DyzArray, SymTen1DxArray, SymTen1DyArray, SymTen1DzArray
export AntiSymTen3DArray, AntiSymTen2DxyArray, AntiSymTen2DxzArray, AntiSymTen2DyzArray

#SOA functions
export tensorarray

#Tensor types inference utilities
export nonzero_eltype
export tensor_type_3D
export tensor_type_2Dxy, tensor_type_2Dxz, tensor_type_2Dyz
export tensor_type_1Dx, tensor_type_1Dy, tensor_type_1Dz

include("type_utils.jl")

"""
    AbstractTensor{N, T} <: AbstractArray{T, N}

Supertype of all Tensor types. Represents any `N`th order tensor with element type `T`.
"""
abstract type AbstractTensor{N, T} <: AbstractArray{T, N} end

# Treat Vec's as scalar when broadcasting
Base.Broadcast.broadcastable(u::AbstractTensor) = (u,)

"""
    Tensor{N, T, Tx, Ty, Tz} <: AbstractTensor{N, T}

A `N`th order tensor with eltype `T`.
`N`th order tensors are implemented as vectors (1st order tensors) of `(N-1)`th order tensors.
A `Tensor` will always have 3 elements, each associated with the `x`,`y`, and `z` directions. Lower dimensional tensors can be constructed by using compile time null values, which are constructed using the `Zero` number from the `Zeros` package.

# Fields
- `x::Tx`: The element associated with the first (x) dimension of the tensor. For a 1st order tensor `t` (a vector) this equivalent to `t[1]`, for a 2nd order tensor (a matrix) `t` this is equivalent to `t[:,1]`, for a 3rd order: `t[:,:,1]`, and so on for higher orders.
- `y::Ty`: The element associated with the second (y) dimension of the tensor. For a 1st order tensor `t` (a vector) this equivalent to `t[2]`, for a 2nd order tensor (a matrix) `t`, this is equivalent to `t[:,2]`, for a 3rd order: `t[:,:,2]`, and so on for higher orders.
- `z::Tz`: The element associated with the third (z) dimension of the tensor. For a 1st order tensor `t` (a vector) this equivalent to `t[3]`, for a 2nd order tensor (a matrix) `t`, this is equivalent to `t[:,3]`, for a 3rd order: `t[:,:,3]`, and so on for higher orders.

# Properties (For `Tensor{N>=2}` only)
- `xx`: Same as `t.x.x`. Ex.g.: For a 3d order tensor `t` this is equivalent to `t[:,1,1]`
- `xy`: Same as `t.y.x`. Ex.g.: For a 3d order tensor `t` this is equivalent to `t[:,1,2]`
- `xz`: Same as `t.z.x`. Ex.g.: For a 3d order tensor `t` this is equivalent to `t[:,1,3]`
- `yx`: Same as `t.x.y`. Ex.g.: For a 3d order tensor `t` this is equivalent to `t[:,2,1]`
- `yy`: Same as `t.y.y`. Ex.g.: For a 3d order tensor `t` this is equivalent to `t[:,2,2]`
- `yz`: Same as `t.z.y`. Ex.g.: For a 3d order tensor `t` this is equivalent to `t[:,2,3]`
- `zx`: Same as `t.x.z`. Ex.g.: For a 3d order tensor `t` this is equivalent to `t[:,3,1]`
- `zy`: Same as `t.y.z`. Ex.g.: For a 3d order tensor `t` this is equivalent to `t[:,3,2]`
- `zz`: Same as `t.z.z`. Ex.g.: For a 3d order tensor `t` this is equivalent to `t[:,3,3]`
"""
struct Tensor{N, T, Tx, Ty, Tz} <: AbstractTensor{N, T}
    x::Tx
    y::Ty
    z::Tz

    @doc """
        Tensor(x, y, z) -> Tensor

    If `x`,`y`,`z` are `Number`s (or of the same type `T` where `T` is not a `AbstractTensor`), returns a vector (`Tensor{1}`) with `x`,`y`,`z` as its elements.
    The elements are promoted to a common type, with the exception of `Zero`s and `One`s, which maintains its type to denote any null or constant direction.

    `x`,`y`,`z` can also be all `AbstractTensor`s of same order `N`, in which case the function returns a `Tensor{N+1}`, and the `eltype`s of `x`,`y`,`z` are promoted to a common type, again with the execption of `Zero`s and `Ones`.

    # Examples
    ```julia-repl
    julia> v1 = Tensor(1,2.,Zero())
    3-element Vec2Dxy{Float64}:
    1.0
    2.0
    𝟎

    julia> v2 = Tensor(3,4,Zero())
    3-element Vec2Dxy{Int64}:
    3
    4
    𝟎

    julia> v3 = Tensor(Zero(),Zero(),Zero())
    3-element Tensor{1, Zero, Zero, Zero, Zero}:
    𝟎
    𝟎
    𝟎

    julia> T = Tensor(v1,v2,v3)
    3×3 Ten2Dxy{Float64}:
    1.0  3.0  𝟎
    2.0  4.0  𝟎
    𝟎    𝟎    𝟎
    ```
    """
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

"""
    Tensor{N}() -> Tensor{N}

Returns a compile time constant null `Tensor` of order `N`.

# Examples
```julia-repl
julia> Tensor{1}()
3-element Tensor{1, Zero, Zero, Zero, Zero}:
 𝟎
 𝟎
 𝟎

julia> Tensor{3}()
3×3×3 Tensor{3, Zero, Tensor{2, Zero, Tensor{1, Zero, Zero, Zero, Zero}, Tensor{1, Zero, Zero, Zero, Zero}, Tensor{1, Zero, Zero, Zero, Zero}}, Tensor{2, Zero, Tensor{1, Zero, Zero, Zero, Zero}, Tensor{1, Zero, Zero, Zero, Zero}, Tensor{1, Zero, Zero, Zero, Zero}}, Tensor{2, Zero, Tensor{1, Zero, Zero, Zero, Zero}, Tensor{1, Zero, Zero, Zero, Zero}, Tensor{1, Zero, Zero, Zero, Zero}}}:
[:, :, 1] =
 𝟎  𝟎  𝟎
 𝟎  𝟎  𝟎
 𝟎  𝟎  𝟎

[:, :, 2] =
 𝟎  𝟎  𝟎
 𝟎  𝟎  𝟎
 𝟎  𝟎  𝟎

[:, :, 3] =
 𝟎  𝟎  𝟎
 𝟎  𝟎  𝟎
 𝟎  𝟎  𝟎
```
"""
@inline Tensor{N}() where N = Tensor(Tensor{N-1}(), Tensor{N-1}(), Tensor{N-1}())

"""
    Tensor(; x = Zero(), y = Zero(), z = Zero()) -> Tensor

Returns a `Tensor` with elements `x`, `y`, `z`. Any unspecified elements are implicitly converted to an appropriate order compile time constant null `Tensor` or scalar.

# Examples
```julia-repl
julia> v1 = Tensor(x=1, y=2.)
3-element Vec2Dxy{Float64}:
 1.0
 2.0
 𝟎

julia> v2 = Tensor(y=4, x=3)
3-element Vec2Dxy{Int64}:
 3
 4
 𝟎

julia> T = Tensor(x=v1, y=v2)
3×3 Ten2Dxy{Float64}:
 1.0  3.0  𝟎
 2.0  4.0  𝟎
 𝟎    𝟎    𝟎
```
"""
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
                return getproperty(getfield(T, :x), :x)
            elseif s === :xy
                return getproperty(getfield(T, :y), :x)
            elseif s === :xz
                return getproperty(getfield(T, :z), :x)
            elseif s === :yx
                return getproperty(getfield(T, :x), :y)
            elseif s === :yy
                return getproperty(getfield(T, :y), :y)
            elseif s === :yz
                return getproperty(getfield(T, :z), :y)
            elseif s === :zx
                return getproperty(getfield(T, :x), :z)
            elseif s === :zy
                return getproperty(getfield(T, :y), :z)
            elseif s === :zz
                return getproperty(getfield(T, :z), :z)
            else
                return getfield(T, s)
            end
        end
    else
        return :(getfield(T,s))
    end
end


# useful compile time constant tensors
const 𝐢 = Vec1Dx(One())
const 𝐣 = Vec1Dy(One())
const 𝐤 = Vec1Dz(One())
const 𝐈 = Tensor(𝐢, 𝐣, 𝐤)
const 𝐢𝐢 = Tensor(𝐢, Vec(), Vec())
const 𝐢𝐣 = Tensor(Vec(), 𝐢, Vec())
const 𝐢𝐤 = Tensor(Vec(), Vec(), 𝐢)
const 𝐣𝐢 = Tensor(𝐣, Vec(), Vec())
const 𝐣𝐣 = Tensor(Vec(), 𝐣, Vec())
const 𝐣𝐤 = Tensor(Vec(), Vec(), 𝐣)
const 𝐤𝐢 = Tensor(𝐤, Vec(), Vec())
const 𝐤𝐣 = Tensor(Vec(), 𝐤, Vec())
const 𝐤𝐤 = Tensor(Vec(), Vec(), 𝐤)

include("operations.jl")

include("symmetric_tensors.jl")

include("antisym_tensors.jl")

include("vecarray.jl")

include("sym_antisym_vecarray.jl")

#### some useful Base methods #######

@inline Base.sum(v::AbstractTensor) = sum(v.x) + sum(v.y) + sum(v.z)

@inline Base.sum(op::F, v::AbstractTensor) where {F <: Function} = @inline sum(op,v.x) + sum(op,v.y) + sum(op,v.z)

@inline Base.map(f::F, vecs::Vararg{AbstractTensor{N}}) where {F <: Function, N} = @inline(Tensor(map(f, _x.(vecs)...), map(f, _y.(vecs)...), map(f, _z.(vecs)...)))

@inline _xx(u::AbstractTensor) = u.xx
@inline _xy(u::AbstractTensor) = u.xy
@inline _xz(u::AbstractTensor) = u.xz
@inline _yy(u::AbstractTensor) = u.yy
@inline _yz(u::AbstractTensor) = u.yz
@inline _zz(u::AbstractTensor) = u.zz

@inline Base.map(f::F, vecs::Vararg{SymmetricTensor{N}}) where {F <: Function, N} = @inline(SymmetricTensor(map(f, _xx.(vecs)...), map(f, _xy.(vecs)...), map(f, _xz.(vecs)...),
                                                                                                            map(f, _yy.(vecs)...), map(f, _yz.(vecs)...), map(f, _zz.(vecs)...)))

#### AbstractMatrix interface defined in the `Base` module #######

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
