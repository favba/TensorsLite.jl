
"""
    AbstractSymmetricTensor{N, T} <: AbstractTensor{N, T}

Represents any `N`th order tensor with element type `T` and some symmetry over its indices.
"""
abstract type AbstractSymmetricTensor{N, T} <: AbstractTensor{N, T} end

"""
    SymTen{T} === AbstractSymmetricTensor{2,T}

Abstract type alias for symmetric matrices (2nd order tensors) with eltype `T`.
"""
const SymTen{T} = AbstractSymmetricTensor{2,T}

"""
    SymmetricTensor{N, T, Txx, Txy, Txz, Tyy, Tyz, Tzz} <: AbstractSymmetricTensor{N, T}

A `N`th (with `N >= 2` always) order tensor with eltype `T` and symmetry over its last two indices.
For example, a `SymmetricTensor{4}` will have the following property: `S[:,:,i,j] == S[:,:,j,i] for i = 1:3, j = 1:3`.
`N`th order symmetric tensors are implemented as matrices (2st order tensors) whose elements are `(N-1)`th order tensors (or scalars, if N == 2).
A `SymmetricTensor` will always have 6 elements, each associated with the `xx`, `xy`, `xz`, `yy`, `yz` and `zz` elements. Lower dimensional tensors can be constructed by using compile time null values, which are constructed using the `Zero` number from the `Zeros` package.

# Fields
- `xx::Txx`: The element associated with the `x` direction of the tensor. For a 2nd order tensor `t` (a matrix) this is equivalent to `t[1,1]`, for a 3rd order: `t[:,1,1]`, and so on for higher orders.
- `xy::Txy`: The element associated with the `xy` plane of the tensor. For a 2nd order tensor `t` (a matrix) this is equivalent to `t[1,2]` or `t[2,1]`, for a 3rd order: `t[:,1,2]` or `t[:,2,1]`, and so on for higher orders.
- `xz::Txz`: The element associated with the `xz` plane of the tensor. For a 2nd order tensor `t` (a matrix) this is equivalent to `t[1,3]` or `t[3,1]`, for a 3rd order: `t[:,1,3]` or `t[:,3,1]`, and so on for higher orders.
- `yy::Tyy`: The element associated with the `y` direction of the tensor. For a 2nd order tensor `t` (a matrix) this is equivalent to `t[2,2]`, for a 3rd order: `t[:,2,2]`, and so on for higher orders.
- `yz::Tyz`: The element associated with the `yz` plane of the tensor. For a 2nd order tensor `t` (a matrix) this is equivalent to `t[2,3]` or `t[3,2]`, for a 3rd order: `t[:,2,3]` or `t[:,3,2]`, and so on for higher orders.
- `zz::Tzz`: The element associated with the `y` direction of the tensor. For a 2nd order tensor `t` (a matrix) this is equivalent to `t[3,3]`, for a 3rd order: `t[:,3,3]`, and so on for higher orders.

# Properties
- `yx`: Same as `t.xy` due to symmetry.
- `zx`: Same as `t.xz` due to symmetry.
- `zy`: Same as `t.yz` due to symmetry.
- `x`: The element associated with the third (x) dimension of the tensor. For a 2nd order tensor (a matrix) `t`, this is equivalent to `t[:,1]`, for a 3rd order: `t[:,:,1]`, and so on for higher orders.
- `y`: The element associated with the third (y) dimension of the tensor. For a 2nd order tensor (a matrix) `t`, this is equivalent to `t[:,2]`, for a 3rd order: `t[:,:,2]`, and so on for higher orders.
- `z`: The element associated with the third (z) dimension of the tensor. For a 2nd order tensor (a matrix) `t`, this is equivalent to `t[:,3]`, for a 3rd order: `t[:,:,3]`, and so on for higher orders.
"""
struct SymmetricTensor{N, T, Txx, Tyx, Tzx, Tyy, Tzy, Tzz} <: AbstractSymmetricTensor{N, T}
    xx::Txx
    xy::Tyx
    xz::Tzx
    yy::Tyy
    yz::Tzy
    zz::Tzz

    @doc """
        SymmetricTensor(xx, xy, xz, yy, yz) -> SymmetricTensor

    If `xx`, `xy`, `xz`, `yy`, `yz`, `zz` are `Numbers` (or of the same type `T` where `T` is not an `AbstractTensor`), returns a symmetric matrix (`SymmetricTensor{2}`) where `yx == xy`, `zx == xz` and `zy == yz`.
    The elements are promoted to a common type, with the exception of `Zero`s and `One`s, which maintains its type to denote any null or constant direction.

    The input variables can also be all any `AbstractTensor`s of same order `N`, in which case the function returns a `SymmetricTensor{N+2}`, and the eltypes of the input are promoted to a common type, again with the execption of `Zero`s and `One`s.

    # Examples
    ```julia-repl
    julia> S = SymmetricTensor(1,2,3,4,5,6)
    3×3 SymTen3D{Int64}:
    1  2  3
    2  4  5
    3  5  6

    julia> S = SymmetricTensor(rand(Vec3D,6)...)
    3×3×3 SymmetricTensor{3, Float64, Vec3D{Float64}, Vec3D{Float64}, Vec3D{Float64}, Vec3D{Float64}, Vec3D{Float64}, Vec3D{Float64}}:
    [:, :, 1] =
    0.0335767  0.977417   0.881416
    0.921797   0.0346131  0.957478
    0.790703   0.256271   0.731106

    [:, :, 2] =
    0.977417   0.584199  0.192881
    0.0346131  0.42396   0.819037
    0.256271   0.610626  0.489607

    [:, :, 3] =
    0.881416  0.192881  0.435205
    0.957478  0.819037  0.0819678
    0.731106  0.489607  0.420909

    ```
    """
    @inline function SymmetricTensor(
            xx, xy, xz,
                yy, yz,
                    zz
        )

        mapreduce(x->isa(x,AbstractTensor), |, (xx,xy,xz,yy,yz,zz)) && throw(DimensionMismatch())

        Txx = typeof(xx)
        Tyx = typeof(xy)
        Tzx = typeof(xz)
        Tyy = typeof(yy)
        Tzy = typeof(yz)
        Tzz = typeof(zz)
        Tf = promote_type_ignoring_Zero_and_One(
            Txx, Tyx, Tzx,
                 Tyy, Tzy,
                      Tzz
        )
        xxn = _my_convert(Tf, xx)
        yxn = _my_convert(Tf, xy)
        zxn = _my_convert(Tf, xz)
        yyn = _my_convert(Tf, yy)
        zyn = _my_convert(Tf, yz)
        zzn = _my_convert(Tf, zz)

        Txxf = typeof(xxn)
        Tyxf = typeof(yxn)
        Tzxf = typeof(zxn)
        Tyyf = typeof(yyn)
        Tzyf = typeof(zyn)
        Tzzf = typeof(zzn)
        Tff = Union{Txxf, Tyxf, Tzxf, Tyyf, Tzyf, Tzzf}
        return new{2, Tff, Txxf, Tyxf, Tzxf, Tyyf, Tzyf, Tzzf}(
            xxn, yxn, zxn,
                 yyn, zyn,
                      zzn
        )
    end

    @inline function SymmetricTensor(
            xx::AbstractTensor{N,Txx}, xy::AbstractTensor{N,Txy}, xz::AbstractTensor{N,Txz},
                yy::AbstractTensor{N,Tyy}, yz::AbstractTensor{N,Tyz},
                    zz::AbstractTensor{N,Tzz}
        ) where {N, Txx, Txy, Txz, Tyy, Tyz, Tzz}

        Tf = promote_type_ignoring_Zero_and_One(map(_non_StaticBool_type,(Txx,Txy,Txz,Tyy,Tyz,Tzz))...)

        xxf = _eltype_convert(Tf,xx)
        xyf = _eltype_convert(Tf,xy)
        xzf = _eltype_convert(Tf,xz)
        yyf = _eltype_convert(Tf,yy)
        yzf = _eltype_convert(Tf,yz)
        zzf = _eltype_convert(Tf,zz)

        return new{
                   N+2,
                   Union{map(eltype,(xxf,xyf,xzf,yyf,yzf,zzf))...},
                   map(typeof,(xxf,xyf,xzf,yyf,yzf,zzf))...
               }(xxf,xyf,xzf,yyf,yzf,zzf)
    end

end

#################################### Aliases ###############################################

"""
    SymTen3D{T} === SymmetricTensor{2, T, T, T, T, T, T, T}

Concrete type alias of 3D 2nd order symmetric tensors with eltype `T`.
"""
const SymTen3D{T} = SymmetricTensor{2, T, T, T, T, T, T, T}

"""
    SymTen2Dxy{T} === Tensor{2, Union{T, Zero}, T, T, Zero, T, Zero, Zero}

Concrete type alias of 2D 2nd order symmetric tensors on the x-y plane with non-null values of type `T`.
"""
const SymTen2Dxy{T} = SymmetricTensor{2, Union{Zero, T}, T, T, Zero, T, Zero, Zero}

"""
    SymTen2Dxz{T} === Tensor{2, Union{T, Zero}, T, Zero, T, Zero, Zero, T}

Concrete type alias of 2D 2nd order symmetric tensors on the x-z plane with non-null values of type `T`.
"""
const SymTen2Dxz{T} = SymmetricTensor{2, Union{Zero, T}, T, Zero, T, Zero, Zero, T}

"""
    SymTen2Dyz{T} === Tensor{2, Union{T, Zero}, Zero, Zero, Zero, T, T, T}

Concrete type alias of 2D 2nd order symmetric tensors on the y-z plane with non-null values of type `T`.
"""
const SymTen2Dyz{T} = SymmetricTensor{2, Union{Zero, T}, Zero, Zero, Zero, T, T, T}

const SymTen2D{T} = Union{SymTen2Dxy{T}, SymTen2Dxz{T}, SymTen2Dyz{T}}

"""
    SymTen1Dx{T} === SymmetricTensor{2, Union{T, Zero}, T, Zero, Zero, Zero, Zero, Zero}

Concrete type alias of 1D 2nd order symmetric tensors on the x direction with a non-null value of type `T`.
"""
const SymTen1Dx{T} = SymmetricTensor{2, Union{Zero, T}, T, Zero, Zero, Zero, Zero, Zero}

"""
    SymTen1Dy{T} === SymmetricTensor{2, Union{T, Zero}, Zero, Zero, Zero, T, Zero, Zero}

Concrete type alias of 1D 2nd order symmetric tensors on the y direction with a non-null value of type `T`.
"""
const SymTen1Dy{T} = SymmetricTensor{2, Union{Zero, T}, Zero, Zero, Zero, T, Zero, Zero}

"""
    SymTen1Dz{T} === SymmetricTensor{2, Union{T, Zero}, Zero, Zero, Zero, Zero, Zero, T}

Concrete type alias of 1D 2nd order symmetric tensors on the z direction with a non-null value of type `T`.
"""
const SymTen1Dz{T} = SymmetricTensor{2, Union{Zero, T}, Zero, Zero, Zero, Zero, Zero, T}

const SymTen1D{T} = Union{SymTen1Dx{T}, SymTen1Dy{T}, SymTen1Dz{T}}

const SymTen0D = SymTen3D{Zero}

#Needed because of ambiguity of type aliases for Ten0D (Ten0D === Ten3D{Zero} === Ten2Dxy{Zero} === Ten1Dx{Zero} === ... === Ten1Dz{Zero})
function Base.show(io::IO, ::Type{SymTen0D})
    print(io, "SymTen0D")
end

const SymTenMaybe2Dxy{T, Tz} = SymmetricTensor{2, Union{T, Tz}, T, T, Tz, T, Tz, Tz}

"""
    DiagSymTen{Txx, Tyy, Tzz} === SymmetricTensor{2, Union{Txx, Tyy, Tzz, Zero}, Txx, Zero, Zero, Tyy, Tzz}

Type alias for diagonal matrices (2nd order tensors) with diagonal elements of types `Txx`, `Tyy` and `Tzz`, respectively.
"""
const DiagSymTen{Txx, Tyy, Tzz} = SymmetricTensor{2, Union{Txx, Tyy, Tzz, Zero}, Txx, Zero, Zero, Tyy, Zero, Tzz}

"""
    DiagSymTen3D{T} === SymmetricTensor{2, Union{Zero,T}, T, Zero, Zero, T, Zero, T}

Concrete type alias of 3D 2nd order diagonal tensors with non-null values of type `T`.
"""
const DiagSymTen3D{T} = DiagSymTen{T, T, T}

"""
    DiagSymTen2Dxy{T} === SymmetricTensor{2, Union{T, Zero}, T, Zero, Zero, T, Zero, Zero}

Concrete type alias of 2D 2nd order diagonal tensors on the x-y plane with non-null values of type `T`.
"""
const DiagSymTen2Dxy{T} = DiagSymTen{T, T, Zero}

"""
    DiagSymTen2Dxz{T} === SymmetricTensor{2, Union{T, Zero}, T, Zero, Zero, Zero, Zero, T}

Concrete type alias of 2D 2nd order diagonal tensors on the x-z plane with non-null values of type `T`.
"""
const DiagSymTen2Dxz{T} = DiagSymTen{T, Zero, T}

"""
    DiagSymTen2Dyz{T} === SymmetricTensor{2, Union{T, Zero}, Zero, Zero, Zero, T, Zero, T}

Concrete type alias of 2D 2nd order diagonal tensors on the y-z plane with non-null values of type `T`.
"""
const DiagSymTen2Dyz{T} = DiagSymTen{Zero, T, T}

#################################### Aliases ###############################################

@inline constructor(::Type{T}) where {T <: SymmetricTensor} = SymmetricTensor

@inline SymmetricTensor{2}() = SymmetricTensor(Zero(),Zero(),Zero(),Zero(),Zero(),Zero())

@inline SymmetricTensor{3}() = SymmetricTensor(Vec(),Vec(),Vec(),Vec(),Vec(),Vec())

"""
    SymmetricTensor{N}() -> SymmetricTensor{N, Zero}

Returns a compile time constant null `SymmetricTensor` of order `N`.
"""
@inline SymmetricTensor{N}() where {N} = SymmetricTensor(SymmetricTensor{N-2}(), SymmetricTensor{N-2}(), SymmetricTensor{N-2}(), SymmetricTensor{N-2}(), SymmetricTensor{N-2}(), SymmetricTensor{N-2}())

"""
    SymmetricTensor(; xx = Zero(), xy = Zero(), xz = Zero(), yy = Zero(), yz = Zero(), zz = Zero()) -> SymmetricTensor

Returns a `SymmetricTensor` with elements `xx`, `xy`, `xz`, `yy`, `yz`, `zz`. Any unspecified elements are implicitly converted to an appropriate order compile time constant null `Tensor` or scalar.

# Examples
```julia-repl
julia> S = SymmetricTensor(xx=1, xy=2, yy=-1.0)
3×3 SymTen2Dxy{Float64}:
 1.0   2.0  𝟎
 2.0  -1.0  𝟎
 𝟎     𝟎    𝟎

julia> S = SymmetricTensor(yy=rand(Vec2Dyz), yz=rand(Vec2Dyz), zz=rand(Vec2Dyz))
3×3×3 SymmetricTensor{3, Union{Zero, Float64}, Vec0D, Vec0D, Vec0D, Vec2Dyz{Float64}, Vec2Dyz{Float64}, Vec2Dyz{Float64}}:
[:, :, 1] =
 𝟎  𝟎  𝟎
 𝟎  𝟎  𝟎
 𝟎  𝟎  𝟎

[:, :, 2] =
 𝟎  𝟎          𝟎
 𝟎  0.534784   0.305258
 𝟎  0.0471851  0.732287

[:, :, 3] =
 𝟎  𝟎         𝟎
 𝟎  0.305258  0.901544
 𝟎  0.732287  0.0543495
"""
@inline function SymmetricTensor(; xx = 𝟎, xy = 𝟎, xz = 𝟎, yy = 𝟎, yz = 𝟎, zz = 𝟎)
    if (xx === 𝟎) && (xy == 𝟎) && (xz == 𝟎) && (yy === 𝟎) && (yz == 𝟎) && (zz == 𝟎)
        return SymmetricTensor{2}()
    else
        check_args_ignoring_zeros(xx,xy,xz,yy,yz,zz)
        NV = _get_ndims(xx,xy,xz,yy,yz,zz)
        xxf, xyf, xzf, yyf, yzf, zzf = if_zero_to_tensor(NV,xx,xy,xz,yy,yz,zz)
        return SymmetricTensor(xxf, xyf, xzf, yyf, yzf, zzf)
    end
end

@inline function Base.convert(::Type{SymmetricTensor{N, T, Txx, Tyx, Tzx, Tyy, Tzy, Tzz}}, v::SymmetricTensor{N}) where {N, T, Txx, Tyx, Tzx, Tyy, Tzy, Tzz}
    @inline nfields = map(convert, (Txx, Tyx, Tzx, Tyy, Tzy, Tzz), fields(v))
    return SymmetricTensor(nfields...)
end

@inline Base.:+(a::SymmetricTensor{N}, b::SymmetricTensor{N}) where {N} = @inline SymmetricTensor(map(+, fields(a), fields(b))...)

@inline Base.:-(a::SymmetricTensor{N}, b::SymmetricTensor{N}) where {N} = @inline SymmetricTensor(map(-, fields(a), fields(b))...)

@inline ==(a::SymmetricTensor{N}, b::SymmetricTensor{N}) where {N} = @inline reduce(&, map(==, fields(a), fields(b)))

@inline function Base.muladd(a::Number, v::SymmetricTensor{N}, u::SymmetricTensor{N}) where {N}
    @inline begin
        at = convert(promote_type(typeof(a), nonzero_eltype(v), nonzero_eltype(u)), a)
        S = SymmetricTensor(map(muladd, ntuple(i -> at, Val(6)), fields(v), fields(u))...)
    end
    return S
end

@inline Base.muladd(::Zero, ::SymmetricTensor{N}, S::SymmetricTensor{N}) where {N} = S

@inline Base.muladd(::One, D::SymmetricTensor{N}, S::SymmetricTensor{N}) where {N} = D+S

"""
    SymTen(xx, xy, xz,
               yy, yz,
                   zz) -> SymmetricTensor{2}

Returns a symmetric matrix (2nd order SymmetricTensor) equivalent to the literal Matrix `[xx xy xz; xy yy yz; xz yz zz]`.
All values are promoted to a common type, with the exception of `Zero`s and `One`s.

# Examples
```julia-repl
julia> S = SymTen(One(), 2, Zero(), 3.0, 4.0, One())
3×3 SymmetricTensor{2, Union{One, Zero, Float64}, One, Float64, Zero, Float64, Float64, One}:
 𝟏    2.0  𝟎
 2.0  3.0  4.0
 𝟎    4.0  𝟏

```
"""
@inline function SymTen(a, b, c, d, e, f)
    if (a isa AbstractTensor || b isa AbstractTensor || c isa AbstractTensor ||
        d isa AbstractTensor || e isa AbstractTensor || f isa AbstractTensor)
        throw(ArgumentError("AbstractTensors are not valid input to the `SymTen` function"))
    end
    return SymmetricTensor(xx=a, xy=b, xz=c, yy=d, yz=e, zz=f)
end

"""
    SymTen(;xx = Zero(), xy = Zero(), xz = Zero(),
                         yy = Zero(), yz = Zero(),
                                      zz = Zero()) -> SymmetricTensor{2}

Returns a symmetric matrix (2nd order SymmetricTensor) equivalent to the literal Matrix `[xx xy xz; xy yy yz; xz yz zz]`.
All values are promoted to a common type, with the exception of `Zero`s and `One`s.

# Examples
```julia-repl
julia> S = SymTen(xx = One(), xy = 2, yy = 3.0, yz = 4.0)
3×3 SymmetricTensor{2, Union{One, Zero, Float64}, One, Float64, Zero, Float64, Float64, Zero}:
 𝟏    2.0  𝟎
 2.0  3.0  4.0
 𝟎    4.0  𝟎

```
"""
@inline SymTen(; xx = 𝟎, xy = 𝟎, xz = 𝟎, yy = 𝟎, yz = 𝟎, zz = 𝟎) = SymTen(xx,xy,xz,yy,yz,zz)

"""
    SymTen3D{T}(xx, xy, xz,
                    yy, yz,
                        zz) -> SymTen3D{T}

Returns a symmetric matrix (2nd order SymmetricTensor) equivalent to the literal Matrix `[xx xy xz; xy yy yz; xz yz zz]` with all values converted to type `T`.

# Examples
```julia-repl
julia> S = SymTen3D{Float32}(Zero(), One(), 1, 2, 3.0, 4.0 + 0.0im)
3×3 SymTen3D{Float32}:
 0.0  1.0  1.0
 1.0  2.0  3.0
 1.0  3.0  4.0

```
"""
SymTen3D{T}(xx, xy, xz, yy, yz, zz) where {T} = SymTen(convert(T, xx), convert(T, xy), convert(T, xz),
                                                                       convert(T, yy), convert(T, yz),
                                                                                       convert(T, zz))

"""
    SymTen3D(xx, xy, xz,
                 yy, yz,
                     zz) -> SymTen3D

Returns a symmetric matrix (2nd order SymmetricTensor) equivalent to the literal Matrix `[xx xy xz; xy yy yz; xz yz zz]` with all values promoted to a common type.

# Examples
```julia-repl
julia> S = SymTen3D(Zero(), One(), 1, 2, 3.0, 4.0 + 0.0im)
3×3 SymTen3D{ComplexF64}:
 0.0+0.0im  1.0+0.0im  1.0+0.0im
 1.0+0.0im  2.0+0.0im  3.0+0.0im
 1.0+0.0im  3.0+0.0im  4.0+0.0im

```
"""
SymTen3D(xx, xy, xz, yy, yz, zz) = SymTen3D{promote_type(typeof(xx), typeof(xy), typeof(xz),
                                                                     typeof(yy), typeof(yz),
                                                                                 typeof(zz))}(xx, xy, xz,
                                                                                                  yy, yz,
                                                                                                      zz)

"""
    SymTen2Dxy{T}(xx, xy,
                      yy) -> SymTen2Dxy{T}

Returns a 2D symmetric matrix (2nd order SymmetricTensor) on the x-y plane equivalent to the literal Matrix `[xx xy 𝟎; xy yy 𝟎; 𝟎 𝟎 𝟎]` with all input values converted to type `T`.

# Examples
```julia-repl
julia> S = SymTen2Dxy{Float32}(One(), 2, Zero())
3×3 SymTen2Dxy{Float32}:
 1.0  2.0  𝟎
 2.0  0.0  𝟎
 𝟎    𝟎    𝟎

```
"""
SymTen2Dxy{T}(xx, xy, yy) where {T} = SymTen(convert(T, xx), convert(T, xy), Zero(),
                                                             convert(T, yy), Zero(),
                                                                             Zero())

"""
    SymTen2Dxy(xx, xy,
                   yy) -> SymTen2Dxy

Returns a 2D symmetric matrix (2nd order SymmetricTensor) on the x-y plane equivalent to the literal Matrix `[xx xy 𝟎; xy yy 𝟎; 𝟎 𝟎 𝟎]` with all input values promoted to a common type.

# Examples
```julia-repl
julia> S = SymTen2Dxy(One(), 2, Zero())
3×3 SymTen2Dxy{Int64}:
 1  2  𝟎
 2  0  𝟎
 𝟎  𝟎  𝟎

```
"""
SymTen2Dxy(xx, xy, yy) = SymTen2Dxy{promote_type(typeof(xx), typeof(xy), typeof(yy))}(xx, xy, yy)

"""
    SymTen2Dxz{T}(xx, xz,
                      zz) -> SymTen2Dxz{T}

Returns a 2D symmetric matrix (2nd order SymmetricTensor) on the x-z plane equivalent to the literal Matrix `[xx 𝟎 xz; 𝟎 𝟎 𝟎; xz 𝟎 zz]` with all input values converted to type `T`.

# Examples
```julia-repl
julia> S = SymTen2Dxz{Float32}(One(), 2, Zero())
3×3 SymTen2Dxz{Float32}:
 1.0  𝟎  2.0
 𝟎    𝟎  𝟎
 2.0  𝟎  0.0

```
"""
SymTen2Dxz{T}(xx, xz, zz) where {T} = SymTen(convert(T, xx), Zero(), convert(T, xz),
                                                             Zero(), Zero(),
                                                                     convert(T, zz))

"""
    SymTen2Dxz(xx, xz,
                   zz) -> SymTen2Dxz

Returns a 2D symmetric matrix (2nd order SymmetricTensor) on the x-z plane equivalent to the literal Matrix `[xx 𝟎 xz; 𝟎 𝟎 𝟎; 𝟎 xz zz]` with all input values promoted to a common type.

# Examples
```julia-repl
julia> S = SymTen2Dxz(One(), 2, Zero())
3×3 SymTen2Dxz{Int64}:
 1  𝟎  2
 𝟎  𝟎  𝟎
 2  𝟎  0

```
"""
SymTen2Dxz(xx, xz, zz) = SymTen2Dxz{promote_type(typeof(xx), typeof(xz), typeof(zz))}(xx, xz, zz)

"""
    SymTen2Dyz{T}(yy, yz,
                      zz) -> SymTen2Dyz{T}

Returns a 2D symmetric matrix (2nd order SymmetricTensor) on the y-z plane equivalent to the literal Matrix `[𝟎 𝟎 𝟎; 𝟎 yy yz; 𝟎 yz zz]` with all input values converted to type `T`.

# Examples
```julia-repl
julia> S = SymTen2Dyz{Float32}(One(), 2, Zero())
3×3 SymTen2Dyz{Float32}:
 𝟎  𝟎    𝟎
 𝟎  1.0  2.0
 𝟎  2.0  0.0

```
"""
SymTen2Dyz{T}(yy, yz, zz) where {T} = SymTen(Zero(), Zero(),         Zero() ,
                                                     convert(T, yy), convert(T, yz),
                                                                     convert(T, zz))

"""
    SymTen2Dyz(yy, yz,
                   zz) -> SymTen2Dyz

Returns a 2D symmetric matrix (2nd order SymmetricTensor) on the y-z plane equivalent to the literal Matrix `[𝟎 𝟎 𝟎; 𝟎 yy yz; 𝟎 yz zz]` with all input values promoted to a common type.

# Examples
```julia-repl
julia> S = SymTen2Dyz(One(), 2, Zero())
3×3 SymTen2Dyz{Int64}:
 𝟎  𝟎  𝟎
 𝟎  1  2
 𝟎  2  0

```
"""
SymTen2Dyz(yy, yz, zz) = SymTen2Dyz{promote_type(typeof(yy), typeof(yz), typeof(zz))}(yy, yz, zz)

"""
    SymTen1Dx{T}(xx) -> SymTen1Dx{T}

Returns a 1D symmetric matrix (2nd order SymmetricTensor) on the x direction with non-null element `xx` converted to type `T`.

# Examples
```julia-repl
julia> SymTen1Dx{Float32}(2)
3×3 SymTen1Dx{Float32}:
 2.0  𝟎  𝟎
 𝟎    𝟎  𝟎
 𝟎    𝟎  𝟎

```
"""
SymTen1Dx{T}(xx) where {T} = SymTen(convert(T, xx), Zero(), Zero(),
                                                    Zero(), Zero(),
                                                            Zero())

"""
    SymTen1Dx(xx) -> SymTen1Dx

Returns a 1D symmetric matrix (2nd order SymmetricTensor) on the x direction with non-null element `xx`.

# Examples
```julia-repl
julia> SymTen1Dx(2)
3×3 SymTen1Dx{Int64}:
 2  𝟎  𝟎
 𝟎  𝟎  𝟎
 𝟎  𝟎  𝟎

```
"""
SymTen1Dx(xx) = SymTen1Dx{typeof(xx)}(xx)

"""
    SymTen1Dy{T}(yy) -> SymTen1Dy{T}

Returns a 1D symmetric matrix (2nd order SymmetricTensor) on the y direction with non-null element `yy` converted to type `T`.

# Examples
```julia-repl
julia> SymTen1Dy{Float32}(2)
3×3 SymTen1Dy{Float32}:
 𝟎  𝟎    𝟎
 𝟎  2.0  𝟎
 𝟎  𝟎    𝟎

```
"""
SymTen1Dy{T}(yy) where {T} = SymTen(Zero(), Zero(),         Zero(),
                                            convert(T, yy), Zero(),
                                                            Zero())

"""
    SymTen1Dy(yy) -> SymTen1Dy

Returns a 1D symmetric matrix (2nd order SymmetricTensor) on the y direction with non-null element `yy`.

# Examples
```julia-repl
julia> SymTen1Dy(2)
3×3 SymTen1Dy{Int64}:
 𝟎  𝟎  𝟎
 𝟎  2  𝟎
 𝟎  𝟎  𝟎

```
"""
SymTen1Dy(yy) = SymTen1Dy{typeof(yy)}(yy)

"""
    SymTen1Dz{T}(zz) -> SymTen1Dz{T}

Returns a 1D symmetric matrix (2nd order SymmetricTensor) on the z direction with non-null element `zz` converted to type `T`.

# Examples
```julia-repl
julia> SymTen1Dz{Float32}(2)
3×3 SymTen1Dz{Float32}:
 𝟎  𝟎  𝟎
 𝟎  𝟎  𝟎
 𝟎  𝟎  2.0

```
"""
SymTen1Dz{T}(zz) where {T} = SymTen(Zero(), Zero(), Zero(),
                                            Zero(), Zero(),
                                                    convert(T, zz))

"""
    SymTen1Dz(zz) -> SymTen1Dz

Returns a 1D symmetric matrix (2nd order SymmetricTensor) on the z direction with non-null element `zz`.

# Examples
```julia-repl
julia> SymTen1Dz(2)
3×3 SymTen1Dz{Int64}:
 𝟎  𝟎  𝟎
 𝟎  𝟎  𝟎
 𝟎  𝟎  2

```
"""
SymTen1Dz(zz) = SymTen1Dz{typeof(zz)}(zz)

# Resolve ambiguities
function SymTen3D{Zero}(a,b,c)
    convert(Zero,a)
    convert(Zero,b)
    convert(Zero,c)
    return SymTen()
end

function SymTen3D{Zero}(a)
    convert(Zero,a)
    return SymTen()
end

Base.@constprop :aggressive function Base.getindex(S::SymmetricTensor{2}, I::Vararg{Integer,2})
    tI = map(Int, I)
    @boundscheck _checkbounds(S, tI...)
    t = (tI[1], tI[2])
    t === (1, 1) && return S.xx
    (t === (2, 1) || t === (1, 2)) && return S.xy
    (t === (3, 1) || t === (1, 3)) && return S.xz
    t === (2, 2) && return S.yy
    (t === (3, 2) || t === (2, 3))  && return S.yz
    return S.zz
end

Base.@constprop :aggressive function Base.getindex(S::SymmetricTensor{N}, I::Vararg{Integer,N}) where {N}
    tI = map(Int, I)
    @boundscheck _checkbounds(S, tI...)
    t = (tI[N-1], tI[N])
    mtI = tI[Base.OneTo(N-2)]
    t === (1, 1) && return @inbounds(S.xx[mtI...])
    (t === (2, 1) || t === (1, 2)) && return @inbounds(S.xy[mtI...])
    (t === (3, 1) || t === (1, 3)) && return @inbounds(S.xz[mtI...])
    t === (2, 2) && return @inbounds(S.yy[mtI...])
    (t === (3, 2) || t === (2, 3))  && return @inbounds(S.yz[mtI...])
    return @inbounds(S.zz[mtI...])
end

@inline function Base.getproperty(S::SymmetricTensor, s::Symbol)
    if s === :x
        xx = getfield(S, :xx)
        yx = getfield(S, :xy)
        zx = getfield(S, :xz)
        return Tensor(xx, yx, zx)
    elseif s === :y
        xy = getfield(S, :xy)
        yy = getfield(S, :yy)
        zy = getfield(S, :yz)
        return Tensor(xy, yy, zy)
    elseif s === :z
        xz = getfield(S, :xz)
        yz = getfield(S, :yz)
        zz = getfield(S, :zz)
        return Tensor(xz, yz, zz)
    elseif s === :yx
        return getfield(S, :xy)
    elseif s === :zx
        return getfield(S, :xz)
    elseif s === :zy
        return getfield(S, :yz)
    else
        return getfield(S, s)
    end
end

@inline inner(a::SymmetricTensor{2,<:Real}, b::SymmetricTensor{2,<:Real}) =  muladd(2, muladd(a.xy, b.xy, muladd(a.xz, b.xz, a.yz*b.yz)), muladd(a.xx, b.xx, muladd(a.yy, b.yy, a.zz*b.zz)))

@inline inneradd(a::SymmetricTensor{2,<:Real}, b::SymmetricTensor{2,<:Real}, c::Real) =  muladd(2, muladd(a.xy, b.xy, muladd(a.xz, b.xz, a.yz*b.yz)), muladd(a.xx, b.xx, muladd(a.yy, b.yy, muladd(a.zz,b.zz, c))))


######################## Especializations ###########################

sym_ten_fields(T::AbstractTensor) = (T.xx, T.xy, T.xz, T.yy, T.yz, T.zz)

@inline Base.:+(a::SymmetricTensor{2}, b::DiagTen) = @inline SymmetricTensor(map(+, sym_ten_fields(a), sym_ten_fields(b))...)
@inline Base.:+(b::DiagTen, a::SymmetricTensor{2}) = @inline SymmetricTensor(map(+, sym_ten_fields(a), sym_ten_fields(b))...)

@inline Base.:-(a::SymmetricTensor{2}, b::DiagTen) = @inline SymmetricTensor(map(-, sym_ten_fields(a), sym_ten_fields(b))...)
@inline Base.:-(b::DiagTen, a::SymmetricTensor{2}) = @inline SymmetricTensor(map(-, sym_ten_fields(b), sym_ten_fields(a))...)

@inline function Base.muladd(a::Number, v::SymmetricTensor{2}, u::DiagTen)
    @inline begin
        at = convert(promote_type(typeof(a), nonzero_eltype(v), nonzero_eltype(u)), a)
        S = SymmetricTensor(map(muladd, ntuple(i -> at, Val(6)), sym_ten_fields(v), sym_ten_fields(u))...)
    end
    return S
end

@inline Base.muladd(::Zero, v::SymmetricTensor{2}, u::DiagTen) = u
@inline Base.muladd(::One, v::SymmetricTensor{2}, u::DiagTen) = v + u

@inline function Base.muladd(a::Number, v::DiagTen, u::SymmetricTensor{2})
    @inline begin
        at = convert(promote_type(typeof(a), nonzero_eltype(v), nonzero_eltype(u)), a)
        S = SymmetricTensor(map(muladd, ntuple(i -> at, Val(6)), sym_ten_fields(v), sym_ten_fields(u))...)
    end
    return S
end

@inline Base.muladd(::Zero, v::DiagTen, u::SymmetricTensor{2}) = u
@inline Base.muladd(::One, v::DiagTen, u::SymmetricTensor{2}) = v + u

@inline otimes(a::Tensor{1}) = SymmetricTensor(a.x*a.x, a.x*a.y, a.x*a.z, a.y*a.y, a.y*a.z, a.z*a.z)

@inline dot(S::SymmetricTensor{2}) = SymTen(dot_upper(S,S)...)

"""
    dott(x::Ten) -> SymTen

Equivalent to `dot(x, transpose(x))`, but returns a `SymmetricTensor{2}`. See [`LinearAlgebra.dot`](@ref).
See also [`tdot`](@ref)
"""
@inline dott(T::AbstractTensor{2}) = SymTen(dot_upper(T, transpose(T))...)

"""
    tdot(x::Ten) -> SymTen

Equivalent to `dot(transpose(x), x)`, but returns a `SymmetricTensor{2}`. See [`LinearAlgebra.dot`](@ref).
See also [`dott`](@ref)
"""
@inline tdot(T::AbstractTensor{2}) = SymTen(dot_upper(transpose(T), T)...)

@inline symmetric(S::SymTen) = S

"""
    symmetric(T::Ten) -> SymTen

Computes the symmetric part of `T`, defined as `(T + transpose(T)) / 2`. See also [`antisymmetric`](@ref)
"""
@inline function symmetric(T::AbstractTensor{2})
    hT = T / 2
    SymTen(map(+, sym_ten_fields(hT), sym_ten_fields(transpose(hT)))...)
end
