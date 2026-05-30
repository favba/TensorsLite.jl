
"""
    AbstractAntiSymmetricTensor{N, T} <: AbstractTensor{N, T}

Represents any `N`th order tensor with element type `T` and anti-symmetry, also known as skew-symmetry, over its last two indices.
"""
abstract type AbstractAntiSymmetricTensor{N,T} <: AbstractTensor{N, T} end

"""
    AntiSymTen{T} === AbstractAntiSymmetricTensor{2,T}

Abstract type alias for antisymmetric matrices (2nd order tensors) with eltype `T`.
"""
const AntiSymTen{T} = AbstractAntiSymmetricTensor{2,Union{T,Zero}}

"""
    AntiSymmetricTensor{N, T, Txy, Txz, Tyz} <: AbstractAntiSymmetricTensor{N, T}

A `N`th (with `N >= 2` always) order tensor with eltype `T` and antisymmetry over its last two indices.
For example, an `AntiSymmetricTensor{4}` will have the following property: `S[:,:,i,j] == -S[:,:,j,i] for i = 1:3, j = 1:3`.
`N`th order antisymmetric tensors are implemented as matrices (2st order tensors) whose elements are `(N-1)`th order tensors (or scalars, if N == 2).
An `AntiSymmetricTensor` will always have 3 elements, each associated with the `xy`, `xz`, and `yz` elements, the diagonal elements are always null. Lower dimensional tensors can be constructed by using compile time null values, which are constructed using the `Zero` number from the `Zeros` package.

# Fields
- `xy::Txy`: The element associated with the `xy` plane of the tensor. For a 2nd order tensor `t` (a matrix) this is equivalent to `t[1,2]` or `-t[2,1]`, for a 3rd order: `t[:,1,2]` or `-t[:,2,1]`, and so on for higher orders.
- `xz::Txz`: The element associated with the `xz` plane of the tensor. For a 2nd order tensor `t` (a matrix) this is equivalent to `t[1,3]` or `-t[3,1]`, for a 3rd order: `t[:,1,3]` or `-t[:,3,1]`, and so on for higher orders.
- `yz::Tyz`: The element associated with the `yz` plane of the tensor. For a 2nd order tensor `t` (a matrix) this is equivalent to `t[2,3]` or `-t[3,2]`, for a 3rd order: `t[:,2,3]` or `-t[:,3,2]`, and so on for higher orders.

# Properties
- `xx`: Zero() for 2nd order tensors, or the appropriate null value for higher order tensors.
- `yy`: Zero() for 2nd order tensors, or the appropriate null value for higher order tensors.
- `zz`: Zero() for 2nd order tensors, or the appropriate null value for higher order tensors.
- `yx`: Same as `-t.xy` due to antisymmetry.
- `zx`: Same as `-t.xz` due to antisymmetry.
- `zy`: Same as `-t.yz` due to antisymmetry.
- `x`: The element associated with the third (x) dimension of the tensor. For a 2nd order tensor (a matrix) `t`, this is equivalent to `t[:,1]`, for a 3rd order: `t[:,:,1]`, and so on for higher orders.
- `y`: The element associated with the third (y) dimension of the tensor. For a 2nd order tensor (a matrix) `t`, this is equivalent to `t[:,2]`, for a 3rd order: `t[:,:,2]`, and so on for higher orders.
- `z`: The element associated with the third (z) dimension of the tensor. For a 2nd order tensor (a matrix) `t`, this is equivalent to `t[:,3]`, for a 3rd order: `t[:,:,3]`, and so on for higher orders.
"""
struct AntiSymmetricTensor{N, T, Tyx, Tzx, Tzy} <: AbstractAntiSymmetricTensor{N, T}
    xy::Tyx
    xz::Tzx
    yz::Tzy

    @doc """
        AntiSymmetricTensor(xy, xz, yz) -> SymmetricTensor

    If `xy`, `xz`, and `yz` are `Numbers` (or of the same type `T` where `T` is not an `AbstractTensor`), returns an antisymmetric matrix (`AntiSymmetricTensor{2}`) where `yx == -xy`, `zx == -xz` and `zy == -yz` and `xx == yy == zz == Zero()`.
    The elements are promoted to a common type, with the exception of `Zero`s, which maintains its type to denote any null direction.

    The input variables can also be all any `AbstractTensor`s of same order `N`, in which case the function returns an `AntiSymmetricTensor{N+2}`, and the eltypes of the input are promoted to a common type, again with the execption of `Zero`s.

    # Examples
    ```julia-repl
    julia> W = AntiSymmetricTensor(One(),2,3)
    3×3 AntiSymTen3D{Int64}:
      𝟎   1  2
     -1   𝟎  3
     -2  -3  𝟎

    julia> W = AntiSymmetricTensor(rand(Vec3D,3)...)
    3×3×3 AntiSymmetricTensor{3, Union{Zero, Float64}, Vec3D{Float64}, Vec3D{Float64}, Vec3D{Float64}}:
    [:, :, 1] =
     𝟎  -0.458681   -0.350075
     𝟎  -0.0518743  -0.727302
     𝟎  -0.412933   -0.189499

    [:, :, 2] =
     0.458681   𝟎  -0.335221
     0.0518743  𝟎  -0.359607
     0.412933   𝟎  -0.38739

    [:, :, 3] =
     0.350075  0.335221  𝟎
     0.727302  0.359607  𝟎
     0.189499  0.38739   𝟎

    ```
    """
    @inline function AntiSymmetricTensor(xy, xz, yz)

        # AbstractTensor are only valid as input when they are all of the same order
        # Same order Tensors as input are dealt with in the next method definition
        xy isa AbstractTensor && throw(DimensionMismatch())
        xz isa AbstractTensor && throw(DimensionMismatch())
        yz isa AbstractTensor && throw(DimensionMismatch())

        Tyx = typeof(xy)
        Tzx = typeof(xz)
        Tzy = typeof(yz)
        mTyx = Base.promote_op(Base.:-, Tyx)
        mTzx = Base.promote_op(Base.:-, Tzx)
        mTzy = Base.promote_op(Base.:-, Tzy)
        Tf = promote_type_ignoring_Zero_and_One(mTyx, mTzx, mTzy)
        yxn = _my_convert_antisym(Tf, xy)
        zxn = _my_convert_antisym(Tf, xz)
        zyn = _my_convert_antisym(Tf, yz)

        Tyxf = typeof(yxn)
        Tzxf = typeof(zxn)
        Tzyf = typeof(zyn)
        Tff = Union{Tyxf, Tzxf, Tzyf}
        return new{2, Union{Tff,Zero}, Tyxf, Tzxf, Tzyf}(yxn, zxn, zyn)
    end

    @inline function AntiSymmetricTensor(xy::AbstractTensor{N,Txy}, xz::AbstractTensor{N,Txz}, yz::AbstractTensor{N,Tyz}) where {N, Txy, Txz, Tyz}
        Tf = promote_type_ignoring_Zero_and_One(_non_StaticBool_type(Txy), _non_StaticBool_type(Txz), _non_StaticBool_type(Tyz))
        xyf = _eltype_convert(Tf, xy)
        xzf = _eltype_convert(Tf, xz)
        yzf = _eltype_convert(Tf, yz)
        return new{N+2, Union{eltype(xyf), eltype(xzf), eltype(yzf),Zero}, typeof(xyf), typeof(xzf), typeof(yzf)}(xyf, xzf, yzf)
    end

end

#################################### Aliases ###############################################

"""
    AntiSymTen3D{T} === AntiSymmetricTensor{2, Union{T, Zero}, T, T, T}

Concrete type alias of 3D 2nd order antisymmetric tensors with non-null values of type `T`.
"""
const AntiSymTen3D{T} = AntiSymmetricTensor{2, Union{T,Zero}, T, T, T}

"""
    AntiSymTen2Dxy{T} === Tensor{2, Union{T, Zero}, T, Zero, Zero}

Concrete type alias of 2D 2nd order antisymmetric tensors on the x-y plane with non-null values of type `T`.
"""
const AntiSymTen2Dxy{T} = AntiSymmetricTensor{2, Union{Zero, T}, T, Zero, Zero}

"""
    AntiSymTen2Dxz{T} === Tensor{2, Union{T, Zero}, Zero, T, Zero}

Concrete type alias of 2D 2nd order antisymmetric tensors on the x-z plane with non-null values of type `T`.
"""
const AntiSymTen2Dxz{T} = AntiSymmetricTensor{2, Union{Zero, T}, Zero, T, Zero}

"""
    AntiSymTen2Dyz{T} === Tensor{2, Union{T, Zero}, Zero, Zero, T}

Concrete type alias of 2D 2nd order antisymmetric tensors on the y-z plane with non-null values of type `T`.
"""
const AntiSymTen2Dyz{T} = AntiSymmetricTensor{2, Union{Zero, T}, Zero, Zero, T}

const AntiSymTen2D{T} = Union{AntiSymTen2Dxy{T},AntiSymTen2Dxz{T},AntiSymTen2Dyz{T}}

const AntiSymTenMaybe2Dxy{T, Tz} = AntiSymmetricTensor{2, Union{T, Zero}, T, Tz, Tz}

const AntiSymTen0D = AntiSymTen3D{Zero}

function Base.show(io::IO, ::Type{AntiSymTen0D})
    print(io, "AntiSymTen0D")
end

#################################### Aliases ###############################################

@inline AntiSymmetricTensor{2}() = AntiSymmetricTensor(Zero(), Zero(), Zero())

@inline AntiSymmetricTensor{3}() = AntiSymmetricTensor(Vec(), Vec(), Vec())

"""
    AntiSymmetricTensor{N}() -> AntiSymmetricTensor{N, Zero}

Returns a compile time constant null `AntiSymmetricTensor` of order `N`.
"""
@inline AntiSymmetricTensor{N}() where {N} = AntiSymmetricTensor(AntiSymmetricTensor{N-2}(), AntiSymmetricTensor{N-2}(), AntiSymmetricTensor{N-2}())

"""
    AntiSymmetricTensor(; xy = Zero(), xz = Zero(), yz = Zero()) -> AntiSymmetricTensor

Returns an `AntiSymmetricTensor` with elements `xy`, `xz`, `yz`. Any unspecified elements are implicitly converted to an appropriate order compile time constant null `Tensor` or scalar.

# Examples
```julia-repl
julia> W = AntiSymmetricTensor(xy=2)
3×3 AntiSymTen2Dxy{Int64}:
  𝟎  2  𝟎
 -2  𝟎  𝟎
  𝟎  𝟎  𝟎

julia> W = AntiSymmetricTensor(yz=rand(Vec2Dyz))
3×3×3 AntiSymmetricTensor{3, Union{Zero, Float64}, Vec0D, Vec0D, Vec2Dyz{Float64}}:
[:, :, 1] =
 𝟎  𝟎  𝟎
 𝟎  𝟎  𝟎
 𝟎  𝟎  𝟎

[:, :, 2] =
 𝟎  𝟎   𝟎
 𝟎  𝟎  -0.649013
 𝟎  𝟎  -0.65254

[:, :, 3] =
 𝟎  𝟎         𝟎
 𝟎  0.649013  𝟎
 𝟎  0.65254   𝟎

"""
@inline function AntiSymmetricTensor(; xy = 𝟎, xz = 𝟎, yz = 𝟎)
    if (xy === 𝟎) && (xz == 𝟎) && (yz == 𝟎)
        return AntiSymmetricTensor(xy,xz,yz)
    else
        check_args_ignoring_zeros(xy,xz,yz)
        NV = _get_ndims(xy,xz,yz)
        xyf, xzf, yzf = if_zero_to_tensor(NV,xy,xz,yz)
        return AntiSymmetricTensor(xyf,xzf,yzf)
    end
end

@inline constructor(::Type{T}) where {T <: AntiSymmetricTensor} = AntiSymmetricTensor

@inline Base.:+(a::AntiSymmetricTensor{N}, b::AntiSymmetricTensor{N}) where {N} = @inline AntiSymmetricTensor(map(+, fields(a), fields(b))...)

@inline Base.:-(a::AntiSymmetricTensor{N}, b::AntiSymmetricTensor{N}) where {N} = @inline AntiSymmetricTensor(map(-, fields(a), fields(b))...)

@inline ==(a::AntiSymmetricTensor{N}, b::AntiSymmetricTensor{N}) where {N} = @inline reduce(&, map(==, fields(a), fields(b)))

@inline function Base.muladd(a::Number, v::AntiSymmetricTensor{N}, u::AntiSymmetricTensor{N}) where {N}
    @inline begin
        at = convert(promote_type(typeof(a), nonzero_eltype(v), nonzero_eltype(u)), a)
        W = AntiSymmetricTensor(map(muladd, ntuple(i -> at, Val(3)), fields(v), fields(u))...)
    end
    return W
end

@inline Base.muladd(::Zero, ::AntiSymmetricTensor{N}, u::AntiSymmetricTensor{N}) where {N} = u

@inline Base.muladd(::One, w::AntiSymmetricTensor{N}, u::AntiSymmetricTensor{N}) where {N} = w+u

"""
    AntiSymTen(xx, xy, xz) -> AntiSymmetricTensor{2}

Returns an antisymmetric matrix (2nd order AntiSymmetricTensor) equivalent to the literal Matrix `[0 xy xz; -xy 0 yz; -xz -yz 0]`.
All values are promoted to a common type, with the exception of `Zero`s.

# Examples
```julia-repl
julia> W = AntiSymTen(One(), Zero(), 3.0)
3×3 AntiSymmetricTensor{2, Union{Zero, Float64}, Float64, Zero, Float64}:
  𝟎     1.0  𝟎
 -1.0   𝟎    3.0
  𝟎    -3.0  𝟎

```
"""
@inline function AntiSymTen(a, b, c)
    if (a isa AbstractTensor || b isa AbstractTensor || c isa AbstractTensor)
        throw(ArgumentError("Tensors are not valid input to the `AntiSymTen` function"))
    end
    return AntiSymmetricTensor(xy=a, xz=b, yz=c)
end

"""
    AntiSymTen(;xy = Zero(), xz = Zero(), yz = Zero()) -> AntiSymmetricTensor{2}

Returns an antisymmetric matrix (2nd order AntiSymmetricTensor) equivalent to the literal Matrix `[0 xy xz; -xy 0 yz; -xz -yz 0]`.
All values are promoted to a common type, with the exception of `Zero`s.

# Examples
```julia-repl
julia> W = AntiSymTen(xy = One(), yz = 4.0)
3×3 AntiSymmetricTensor{2, Union{Zero, Float64}, Float64, Zero, Float64}:
  𝟎     1.0  𝟎
 -1.0   𝟎    4.0
  𝟎    -4.0  𝟎

```
"""
@inline AntiSymTen(; xy = 𝟎, xz = 𝟎, yz = 𝟎) = AntiSymTen(xy,xz,yz)

"""
    AntiSymTen3D{T}(xy, xz, yz) -> AntiSymTen3D{T}

Returns an antisymmetric matrix (2nd order AntiSymmetricTensor) equivalent to the literal Matrix `[0 xy xz; -xy 0 yz; -xz -yz 0]` with all values converted to type `T`.

# Examples
```julia-repl
julia> W = AntiSymTen3D{Float32}(Zero(), One(), 4.0 + 0.0im)
3×3 AntiSymTen3D{Float32}:
  𝟎     0.0  1.0
 -0.0   𝟎    4.0
 -1.0  -4.0  𝟎

```
"""
AntiSymTen3D{T}(xy, xz, yz) where {T} = AntiSymTen(convert(T, xy), convert(T, xz), convert(T, yz))

"""
    AntiSymTen3D(xy, xz, yz) -> AntiSymTen3D

Returns an antisymmetric matrix (2nd order AntiSymmetricTensor) equivalent to the literal Matrix `[0 xy xz; -xy 0 yz; -xz -yz 0]` with all values promoted to a common type.

# Examples
```julia-repl
julia> S = AntiSymTen3D(Zero(), One(), 4.0 + 0.0im)
3×3 AntiSymTen3D{ComplexF64}:
     𝟎        0.0+0.0im  1.0+0.0im
 -0.0-0.0im      𝟎       4.0+0.0im
 -1.0-0.0im  -4.0-0.0im     𝟎

```
"""
AntiSymTen3D(xy, xz, yz) = AntiSymTen3D{promote_type(typeof(xy), typeof(xz), typeof(yz))}(xy, xz, yz)


"""
    AntiSymTen2Dxy{T}(xy) -> AntiSymTen2Dxy{T}

Returns a 2D antisymmetric matrix (2nd order AntiSymmetricTensor) on the x-y plane equivalent to the literal Matrix `[𝟎 xy 𝟎; -xy 𝟎 𝟎; 𝟎 𝟎 𝟎]` with all input values converted to type `T`.

# Examples
```julia-repl
julia> W = AntiSymTen2Dxy{Float32}(2)
3×3 AntiSymTen2Dxy{Float32}:
  𝟎    2.0  𝟎
 -2.0  𝟎    𝟎
  𝟎    𝟎    𝟎

```
"""
AntiSymTen2Dxy{T}(xy) where {T} = AntiSymTen(convert(T, xy), Zero(), Zero())

"""
    AntiSymTen2Dxy(xy) -> AntiSymTen2Dxy{typeof(-xy)}

Returns a 2D antisymmetric matrix (2nd order AntiSymmetricTensor) on the x-y plane equivalent to the literal Matrix `[𝟎 xy 𝟎; -xy 𝟎 𝟎; 𝟎 𝟎 𝟎]`.

# Examples
```julia-repl
julia> W = AntiSymTen2Dxy(true)
3×3 AntiSymTen2Dxy{Int64}:
  𝟎  1  𝟎
 -1  𝟎  𝟎
  𝟎  𝟎  𝟎

```
"""
AntiSymTen2Dxy(xy) = AntiSymTen2Dxy{typeof(xy)}(xy)


"""
    AntiSymTen2Dxz{T}(xz) -> AntiSymTen2Dxz{T}

Returns a 2D symmetric matrix (2nd order AntiSymmetricTensor) on the x-z plane equivalent to the literal Matrix `[𝟎 𝟎 xz; 𝟎 𝟎 𝟎; -xz 𝟎 𝟎]` with all input values converted to type `T`.

# Examples
```julia-repl
julia> W = AntiSymTen2Dxz{Float32}(2)
3×3 AntiSymTen2Dxz{Float32}:
  𝟎    𝟎  2.0
  𝟎    𝟎  𝟎
 -2.0  𝟎  𝟎

```
"""
AntiSymTen2Dxz{T}(xz) where {T} = AntiSymTen(Zero(), convert(T, xz), Zero())

"""
    AntiSymTen2Dxz(xz) -> AntiSymTen2Dxz{typeof(-xz)}

Returns a 2D antisymmetric matrix (2nd order AntiSymmetricTensor) on the x-y plane equivalent to the literal Matrix `[𝟎 𝟎 xz; 𝟎 𝟎 𝟎; -xz 𝟎 𝟎]`.

# Examples
```julia-repl
julia> W = AntiSymTen2Dxz(true)
3×3 AntiSymTen2Dxz{Int64}:
  𝟎  𝟎  1
  𝟎  𝟎  𝟎
 -1  𝟎  𝟎

```
"""
AntiSymTen2Dxz(xz) = AntiSymTen2Dxz{typeof(xz)}(xz)


"""
    AntiSymTen2Dyz{T}(yz) -> AntiSymTen2Dyz{T}

Returns a 2D antisymmetric matrix (2nd order AntiSymmetricTensor) on the y-z plane equivalent to the literal Matrix `[𝟎 𝟎 𝟎; 𝟎 𝟎 yz; 𝟎 -yz 𝟎]` with all input values converted to type `T`.

# Examples
```julia-repl
julia> W = AntiSymTen2Dyz{Float32}(2)
3×3 AntiSymTen2Dyz{Float32}:
 𝟎   𝟎    𝟎
 𝟎   𝟎    2.0
 𝟎  -2.0  𝟎

```
"""
AntiSymTen2Dyz{T}(yz) where {T} = AntiSymTen(Zero(), Zero(), convert(T, yz))

"""
    AntiSymTen2Dyz(yz) -> AntiSymTen2Dyz{typeof(-yz)}

Returns a 2D antisymmetric matrix (2nd order AntiSymmetricTensor) on the x-y plane equivalent to the literal Matrix `[𝟎 𝟎 𝟎; 𝟎 𝟎 yz; 𝟎 -yz 𝟎]`.

# Examples
```julia-repl
julia> W = AntiSymTen2Dyz(true)
3×3 AntiSymTen2Dyz{Int64}:
 𝟎   𝟎  𝟎
 𝟎   𝟎  1
 𝟎  -1  𝟎

```
"""
AntiSymTen2Dyz(yz) = AntiSymTen2Dyz{typeof(yz)}(yz)

function AntiSymTen3D{Zero}(a)
    convert(Zero,a)
    return AntiSymTen()
end

Base.@constprop :aggressive function Base.getindex(S::AntiSymmetricTensor{2}, I::Vararg{Integer,2})

    tI = map(Int, I)
    @boundscheck _checkbounds(S, tI...)
    t = (tI[1], tI[2])

    t === (2, 1) && return -S.xy
    t === (3, 1) && return -S.xz
    t === (1, 2) && return S.xy
    t === (3, 2) && return -S.yz
    t === (1, 3) && return S.xz
    t === (2, 3) && return S.yz
    # if (t === (1,1) || t === (2,2) || t === (3,3))
    return 𝟎
end

Base.@constprop :aggressive function Base.getindex(S::AntiSymmetricTensor{N}, I::Vararg{Integer,N}) where {N}

    tI = map(Int, I)
    @boundscheck _checkbounds(S, tI...)
    t = (tI[N-1], tI[N])
    mtI = tI[Base.OneTo(N-2)]

    t === (2, 1) && return -@inbounds(S.xy[mtI...])
    t === (3, 1) && return -@inbounds(S.xz[mtI...])
    t === (1, 2) && return @inbounds(S.xy[mtI...])
    t === (3, 2) && return -@inbounds(S.yz[mtI...])
    t === (1, 3) && return @inbounds(S.xz[mtI...])
    t === (2, 3) && return @inbounds(S.yz[mtI...])
    # if (t === (1,1) || t === (2,2) || t === (3,3))
    return 𝟎
end

@inline function Base.getproperty(S::AntiSymmetricTensor{N}, s::Symbol) where {N}
    zt = Tensor{N-2}()
    if s === :x
        xx = zt
        yx = -getfield(S, :xy)
        zx = -getfield(S, :xz)
        return Tensor(xx, yx, zx)
    elseif s === :y
        xy = getfield(S, :xy)
        yy = zt
        zy = -getfield(S, :yz)
        return Tensor(xy, yy, zy)
    elseif s === :z
        xz = getfield(S, :xz)
        yz = getfield(S, :yz)
        zz = zt
        return Tensor(xz, yz, zz)
    elseif s === :yx
        return -getfield(S, :xy)
    elseif s === :zx
        return -getfield(S, :xz)
    elseif s === :zy
        return -getfield(S, :yz)
    elseif s === :xx || s === :yy || s === :zz
        return zt
    else
        return getfield(S, s)
    end
end

@inline function Base.convert(::Type{AntiSymmetricTensor{N, T, Tyx, Tzx, Tzy}}, v::AntiSymmetricTensor{N}) where {N, T, Tyx, Tzx, Tzy}
    @inline nfields = map(convert, (Tyx, Tzx, Tzy), fields(v))
    return AntiSymmetricTensor(nfields...)
end

################# Especializations ################################

@inline antisym_ten_fields(T::AbstractTensor) = (T.xy, T.xz, T.yz)

@inline inner(a::AntiSymmetricTensor{2, <:Real}, b::AntiSymmetricTensor{2, <:Real}) = 2 * muladd(a.xy, b.xy, muladd(a.xz, b.xz, a.yz * b.yz))

@inline inneradd(a::AntiSymmetricTensor{2, <:Real}, b::AntiSymmetricTensor{2, <:Real}, c::Real) = muladd(2, muladd(a.xy, b.xy, muladd(a.xz, b.xz, a.yz * b.yz)), c)

@inline inner(::AntiSymmetricTensor{2}, ::SymmetricTensor{2}) = 𝟎

@inline inner(::SymmetricTensor{2}, ::AntiSymmetricTensor{2}) = 𝟎

@inline inneradd(::AntiSymmetricTensor{2}, ::SymmetricTensor{2}, c::Number) = c

@inline inneradd(::SymmetricTensor{2}, ::AntiSymmetricTensor{2}, c::Number) = c

@inline dot(W::AntiSymmetricTensor{2}) = SymTen(dot_upper(W,W)...)

@inline symmetric(::AntiSymTen) = SymTen()

@inline antisymmetric(W::AntiSymTen) = W

@inline antisymmetric(::SymTen) = AntiSymTen()

"""
    antisymmetric(T::Ten) -> AntiSymTen

Computes the antisymmetric part of `T`, defined as `(T - transpose(T)) / 2`. See also [`symmetric`](@ref)
"""
@inline function antisymmetric(T::Ten)
    Tu = antisym_ten_fields(T) ./ 2
    Td = antisym_ten_fields(transpose(T)) ./ 2
    AntiSymTen(map(-, Tu, Td)...)
end
