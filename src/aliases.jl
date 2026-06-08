
########################## aliases ###########################

"""
    Vec{T} === AbstractTensor{1,T}

Abstract type alias for vectors (1st order tensors) with eltype `T`.
"""
const Vec{T} = AbstractTensor{1, T}

"""
    Vec3D{T} === Tensor{1, T, T, T, T}

Concrete type alias of 3D vectors with eltype `T`.
"""
const Vec3D{T} = Tensor{1,T,T,T,T}

"""
    Vec2Dxy{T} === Tensor{1, Union{T, Zero}, T, T, Zero}

Concrete type alias of 2D vectors on the x-y plane with non-null values of type `T`.
"""
const Vec2Dxy{T} = Tensor{1, Union{Zero, T}, T, T, Zero}

const DVec2Dxy = Tensor{1, <:Any, <:Any, <:Any, Zero}

"""
    Vec2Dxz{T} === Tensor{1, Union{T, Zero}, T, Zero, T}

Concrete type alias of 2D vectors on the x-z plane with non-null values of type `T`.
"""
const Vec2Dxz{T} = Tensor{1, Union{Zero, T}, T, Zero, T}

const DVec2Dxz = Tensor{1, <:Any, <:Any, Zero, <:Any}

"""
    Vec2Dyz{T} === Tensor{1, Union{T, Zero}, Zero, T, T}

Concrete type alias of 2D vectors on the y-z plane with non-null values of type `T`.
"""
const Vec2Dyz{T} = Tensor{1, Union{Zero, T}, Zero, T, T}

const DVec2Dyz = Tensor{1, <:Any, Zero, <:Any, <:Any}

const Vec2D{T} = Union{Vec2Dxy{T}, Vec2Dxz{T}, Vec2Dyz{T}}

"""
    Vec1Dx{T} === Tensor{1, Union{T, Zero}, T, Zero, Zero}

Concrete type alias of 1D vectors on the x direction with non-null value of type `T`.
"""
const Vec1Dx{T} = Tensor{1, Union{Zero, T}, T, Zero, Zero}

const DVec1Dx = Tensor{1, <:Any, <:Any, Zero, Zero}

"""
    Vec1Dy{T} === Tensor{1, Union{T, Zero}, Zero, T, Zero}

Concrete type alias of 1D vectors on the y direction with non-null value of type `T`.
"""
const Vec1Dy{T} = Tensor{1, Union{Zero, T}, Zero, T, Zero}

const DVec1Dy = Tensor{1, <:Any, Zero, <:Any, Zero}

"""
    Vec1Dz{T} === Tensor{1, Union{T, Zero}, Zero, Zero, T}

Concrete type alias of 1D vectors on the z direction with non-null value of type `T`.
"""
const Vec1Dz{T} = Tensor{1, Union{Zero, T}, Zero, Zero, T}

const DVec1Dz = Tensor{1, <:Any, Zero, Zero, <:Any}

const Vec1D{T} = Union{Vec1Dx{T}, Vec1Dy{T}, Vec1Dz{T}}
const VecND{T} = Union{Vec3D{T}, Vec2D{T}, Vec1D{T}}

const Vec0D = Vec3D{Zero}

#Needed because of ambiguity of type aliases for Vec0D (Vec0D === Vec3D{Zero} === Vec2Dxy{Zero} === Vec1Dx{Zero} === ... === Vec1Dz{Zero})
function Base.show(io::IO, ::Type{Vec0D})
    print(io, "Vec0D")
end

# This ends up also being a VecMaybe1Dz{T, Tz}, if T===Zero and Tz != Zero
const VecMaybe2Dxy{T, Tz} = Tensor{1, Union{T, Tz}, T, T, Tz}

"""
    Ten{T} === AbstractTensor{2,T}

Abstract type alias for matrices (2nd order tensors) with eltype `T`.
"""
const Ten{T} = AbstractTensor{2, T}

"""
    Ten3D{T} === Tensor{2, T, Vec3D{T}, Vec3D{T}, Vec3D{T}}

Concrete type alias of 3D 2nd order tensors with eltype `T`.
"""
const Ten3D{T} = Tensor{2,T,Vec3D{T},Vec3D{T},Vec3D{T}}

"""
    Ten2Dxy{T} === Tensor{2, Union{T, Zero}, Vec3D{T}, Vec3D{T}, Vec3D{Zero}}

Concrete type alias of 2D 2nd order tensors on the x-y plane with non-null values of type `T`.
"""
const Ten2Dxy{T} = Tensor{2, Union{Zero, T}, Vec2Dxy{T}, Vec2Dxy{T}, Vec0D}

const DTen2Dxy = Tensor{2, <:Any, <:DVec2Dxy, <:DVec2Dxy, Vec0D}

"""
    Ten2Dxz{T} === Tensor{2, Union{T, Zero}, Vec2Dxy{T}, Vec2Dxy{Zero}, Vec2Dxy{T}}

Concrete type alias of 2D 2nd order tensors on the x-z plane with non-null values of type `T`.
"""
const Ten2Dxz{T} = Tensor{2, Union{Zero, T}, Vec2Dxz{T}, Vec0D, Vec2Dxz{T}}

const DTen2Dxz = Tensor{2, <:Any, <:DVec2Dxz, Vec0D, <:DVec2Dxz}

"""
    Ten2Dyz{T} === Tensor{2, Union{T, Zero}, Vec2Dyz{Zero}, Vec2Dyz{T}, Vec2Dyz{T}}

Concrete type alias of 2D 2nd order tensors on the y-z plane with non-null values of type `T`.
"""
const Ten2Dyz{T} = Tensor{2, Union{Zero, T}, Vec0D, Vec2Dyz{T}, Vec2Dyz{T}}

const DTen2Dyz = Tensor{2, <:Any, Vec0D, <:DVec2Dyz, <:DVec2Dyz}

const Ten2D{T} = Union{Ten2Dxy{T}, Ten2Dxz{T}, Ten2Dyz{T}}

"""
    Ten1Dx{T} === Tensor{2, Union{T, Zero}, Vec1Dx{T}, Vec1Dx{Zero}, Vec1Dx{Zero}}

Concrete type alias of 1D 2nd order tensors on the x direction with a non-null value of type `T`.
"""
const Ten1Dx{T} = Tensor{2, Union{Zero, T}, Vec1Dx{T}, Vec0D, Vec0D}

const DTen1Dx = Tensor{2, <:Any, <:DVec1Dx, Vec0D, Vec0D}

"""
    Ten1Dy{T} === Tensor{2, Union{T, Zero}, Vec1Dy{Zero}, Vec1Dy{T}, Vec1Dy{Zero}}

Concrete type alias of 1D 2nd order tensors on the y direction with a non-null value of type `T`.
"""
const Ten1Dy{T} = Tensor{2, Union{Zero, T}, Vec0D, Vec1Dy{T}, Vec0D}

const DTen1Dy = Tensor{2, <:Any, Vec0D, <:DVec1Dy, Vec0D}

"""
    Ten1Dz{T} === Tensor{2, Union{T, Zero}, Vec1Dz{Zero}, Vec1Dz{Zero}, Vec1Dz{T}}

Concrete type alias of 1D 2nd order tensors on the z direction with a non-null value of type `T`.
"""
const Ten1Dz{T} = Tensor{2, Union{Zero, T}, Vec0D, Vec0D, Vec1Dz{T}}

const DTen1Dz = Tensor{2, <:Any, Vec0D, Vec0D, <:DVec1Dz}

const Ten1D{T} = Union{Ten1Dx{T}, Ten1Dy{T}, Ten1Dz{T}}
const TenND{T} = Union{Ten3D{T}, Ten2D{T}, Ten1D{T}}
const Ten0D = Ten3D{Zero}

#Needed because of ambiguity of type aliases for Ten0D (Ten0D === Ten3D{Zero} === Ten2Dxy{Zero} === Ten1Dx{Zero} === ... === Ten1Dz{Zero})
function Base.show(io::IO, ::Type{Ten0D})
    print(io, "Ten0D")
end

# I'm not exporting it only because it breaks pretty printing of 1D Tensor types.
"""
    DiagTen{Txx, Tyy, Tzz} === Tensor{2, Union{Txx, Tyy, Tzz, Zero}, Vec1Dx{Txx}, Vec1Dy{Tyy}, Vec1Dz{Tzz}}

Type alias for diagonal matrices (2nd order tensors) with diagonal elements of types `Txx`, `Tyy` and `Tzz`, respectively.
"""
const DiagTen{Txx, Tyy, Tzz} = Tensor{2, Union{Txx, Tyy, Tzz, Zero}, Vec1Dx{Txx}, Vec1Dy{Tyy}, Vec1Dz{Tzz}}

const DDiagTen = Tensor{2, <:Any, <:DVec1Dx, <:DVec1Dy, <:DVec1Dz}

"""
    DiagTen3D{T} === Tensor{2, Union{T, Zero}, Vec1Dx{T}, Vec1Dy{T}, Vec1Dz{T}}

Concrete type alias of 3D 2nd order diagonal tensors with non-null values of type `T`.
"""
const DiagTen3D{T} = DiagTen{T,T,T}

"""
    DiagTen2Dxy{T} === Tensor{2, Union{T, Zero}, Vec1Dx{T}, Vec1Dy{T}, Vec1Dz{Zero}}

Concrete type alias of 2D 2nd order diagonal tensors on the x-y plane with non-null values of type `T`.
"""
const DiagTen2Dxy{T} = DiagTen{T, T, Zero}

"""
    DiagTen2Dxz{T} === Tensor{2, Union{T, Zero}, Vec1Dx{T}, Vec1Dy{Zero}, Vec1Dz{T}}

Concrete type alias of 2D 2nd order diagonal tensors on the x-z plane with non-null values of type `T`.
"""
const DiagTen2Dxz{T} = DiagTen{T, Zero, T}

"""
    DiagTen2Dyz{T} === Tensor{2, Union{T, Zero}, Vec1Dx{Zero}, Vec1Dy{T}, Vec1Dz{T}}

Concrete type alias of 2D 2nd order diagonal tensors on the y-z plane with non-null values of type `T`.
"""
const DiagTen2Dyz{T} = DiagTen{Zero, T, T}

const TenMaybe2Dxy{T, Tz} = Tensor{2, Union{T, Tz}, VecMaybe2Dxy{T, Tz}, VecMaybe2Dxy{T, Tz}, Vec3D{Tz}}

#Useful for dispatch of inv, eigen, etc
const QuasiTen2Dxy = Tensor{2, <:Any, <:DVec2Dxy, <:DVec2Dxy, <:DVec1Dz}
const QuasiTen2Dxz = Tensor{2, <:Any, <:DVec2Dxz, <:DVec1Dy, <:DVec2Dxz}
const QuasiTen2Dyz = Tensor{2, <:Any, <:DVec1Dx, <:DVec2Dyz, <:DVec2Dyz}

########################## aliases ###########################

########################## Vec constructors ###########################

"""
    Vec(x,y,z) -> Tensor{1}

Returns a vector (1st order tensor) with `x`, `y`, `z` as its elements.
The elements are promoted to a common type, with the exception of `Zero`s and `One`s, which maintains its type to denote any null or constant direction.
Throws an error if any of the input values is not a scalar.

# Examples
```julia-repl
julia> v1 = Vec(1,2.,Zero())
3-element Vec2Dxy{Float64}:
 1.0
 2.0
 𝟎

julia> v2 = Vec(3,4,Zero())
3-element Vec2Dxy{Int64}:
 3
 4
 𝟎

julia> T = Vec(v1,v2,v2)
ERROR: ArgumentError: Tensors are not valid input to the `Vec` function
Stacktrace:
 [1] (Vec)(a::Vec2Dxy{Float64}, b::Vec2Dxy{Int64}, c::Vec2Dxy{Int64})

```
"""
@inline function Vec(a, b, c)
    if (a isa AbstractTensor || b isa AbstractTensor || c isa AbstractTensor)
        throw(ArgumentError("Tensors are not valid input to the `Vec` function"))
    end
    return Tensor(a, b, c)
end

"""
    Vec(; x = Zero(), y = Zero(), z = Zero()) -> Tensor{1}

Returns a vector with elements `x`, `y`, `z`, promoting the values to a common type, with the exception of `Zero`s and `One`s.
Throws an error if any of the input values is not a scalar.

# Examples
```julia-repl
julia> v1 = Vec(x=1, y=2.)
3-element Vec2Dxy{Float64}:
 1.0
 2.0
 𝟎

julia> v2 = Vec(y=4, x=3)
3-element Vec2Dxy{Int64}:
 3
 4
 𝟎

julia> T = Vec(x=v1, y=v2)
ERROR: ArgumentError: Tensors are not valid input to the `Vec` function

```
"""
@inline Vec(;x=𝟎,y=𝟎,z=𝟎) = Vec(x,y,z)

"""
    Vec3D{T}(x, y, z) -> Vec3D{T}

Returns a 3D vector with elements `x`,`y`,`z` of type `T`.

# Examples
```julia-repl
julia> Vec3D{Float32}(1,2,3)
3-element Vec3D{Float32}:
 1.0
 2.0
 3.0

julia> Vec3D{ComplexF32}(Zero(), One(), 2)
3-element Vec3D{ComplexF32}:
 0.0f0 + 0.0f0im
 1.0f0 + 0.0f0im
 2.0f0 + 0.0f0im

```
"""
@inline Vec3D{T}(a, b, c) where {T} = Vec(convert(T, a), convert(T, b), convert(T, c))

"""
    Vec3D(x,y,z) -> Vec3D

Returns a 3D vector (1st order tensor) with `x`, `y`, `z` promoted to a common type.

# Examples
```julia-repl
julia> Vec3D(Zero(), One(), 2+0im)
3-element Vec3D{Complex{Int64}}:
 0 + 0im
 1 + 0im
 2 + 0im

```
"""
@inline Vec3D(a::T1, b::T2, c::T3) where {T1,T2,T3} = Vec3D{promote_type(T1,T2,T3)}(a, b, c)

"""
    Vec2Dxy{T}(x, y) -> Vec2Dxy{T}

Returns a 2D vector on the x-y plane with non-null elements `x`,`y` of type `T`.

# Examples
```julia-repl
julia> Vec2Dxy{Float32}(1,2)
3-element Vec2Dxy{Float32}:
 1.0f0
 2.0f0
 𝟎

julia> Vec2Dxy{ComplexF32}(Zero(), One())
3-element Vec2Dxy{ComplexF32}:
 0.0f0 + 0.0f0im
 1.0f0 + 0.0f0im
       𝟎

```
"""
@inline Vec2Dxy{T}(a, b) where {T} = Vec(convert(T, a), convert(T, b), Zero())

"""
    Vec2Dxy(x, y) -> Vec2Dxy

Returns a 2D vector on the x-y plane with non-null elements `x`,`y` promoted to a common type.

# Examples
```julia-repl
julia> Vec2Dxy(1, 2.)
3-element Vec2Dxy{Float64}:
 1.0
 2.0
 𝟎

julia> Vec2Dxy(Zero(), One())
3-element Vec2Dxy{Bool}:
 false
  true
     𝟎

```
"""
@inline Vec2Dxy(a, b) = Vec2Dxy{promote_type(typeof(a),typeof(b))}(a, b)

"""
    Vec2Dxz{T}(x, z) -> Vec2Dxz{T}

Returns a 2D vector on the x-z plane with non-null elements `x`,`z` of type `T`.

# Examples
```julia-repl
julia> Vec2Dxz{Float32}(1,2)
3-element Vec2Dxz{Float32}:
 1.0f0
 𝟎
 2.0f0

julia> Vec2Dxz{ComplexF32}(Zero(), One())
3-element Vec2Dxz{ComplexF32}:
 0.0f0 + 0.0f0im
       𝟎
 1.0f0 + 0.0f0im

```
"""
@inline Vec2Dxz{T}(a, b) where {T} = Vec(convert(T, a), Zero(), convert(T, b))

"""
    Vec2Dxz(x, z) -> Vec2Dxz

Returns a 2D vector on the x-z plane with non-null elements `x`,`z` promoted to a common type.

# Examples
```julia-repl
julia> Vec2Dxz(1, 2.)
3-element Vec2Dxz{Float64}:
 1.0
 𝟎
 2.0

julia> Vec2Dxz(Zero(), One())
3-element Vec2Dxz{Bool}:
 false
     𝟎
  true

```
"""
@inline Vec2Dxz(a, b) = Vec2Dxz{promote_type(typeof(a),typeof(b))}(a, b)

"""
    Vec2Dyz{T}(y, z) -> Vec2Dyz{T}

Returns a 2D vector on the y-z plane with non-null elements `y`,`z` of type `T`.

# Examples
```julia-repl
julia> Vec2Dyz{Float32}(1,2)
3-element Vec2Dyz{Float32}:
 𝟎
 1.0f0
 2.0f0

julia> Vec2Dyz{ComplexF32}(Zero(), One())
3-element Vec2Dyz{ComplexF32}:
       𝟎
 0.0f0 + 0.0f0im
 1.0f0 + 0.0f0im

```
"""
@inline Vec2Dyz{T}(a, b) where {T} = Vec(Zero(), convert(T, a), convert(T, b))

"""
    Vec2Dyz(y, z) -> Vec2Dyz

Returns a 2D vector on the y-z plane with non-null elements `y`,`z` promoted to a common type.

# Examples
```julia-repl
julia> Vec2Dyz(1, 2.)
3-element Vec2Dyz{Float64}:
 𝟎
 1.0
 2.0

julia> Vec2Dyz(Zero(), One())
3-element Vec2Dyz{Bool}:
     𝟎
 false
  true

```
"""
@inline Vec2Dyz(a, b) = Vec2Dyz{promote_type(typeof(a),typeof(b))}(a, b)

"""
    Vec1Dx{T}(x) -> Vec1Dx{T}

Returns a 1D vector on the x direction with non-null element `x` converted to type `T`.

# Examples
```julia-repl
julia> Vec1Dx{Float32}(2)
3-element Vec1Dx{Float32}:
 2.0f0
 𝟎
 𝟎

julia> Vec1Dx{ComplexF32}(Zero())
3-element Vec1Dx{ComplexF32}:
 0.0f0 + 0.0f0im
       𝟎
       𝟎

```
"""
@inline Vec1Dx{T}(a) where {T} = Vec(convert(T, a), Zero(), Zero())

"""
    Vec1Dx(x) -> Vec1Dx

Returns a 1D vector on the x direction with non-null element `x`.

# Examples
```julia-repl
julia> Vec1Dx(2.)
3-element Vec1Dx{Float64}:
 2.0
 𝟎
 𝟎

julia> Vec1Dx(One())
3-element Vec1Dx{One}:
 𝟏
 𝟎
 𝟎

```
"""
@inline Vec1Dx(a) = Vec1Dx{typeof(a)}(a)

"""
    Vec1Dy{T}(y) -> Vec1Dy{T}

Returns a 1D vector on the y direction with non-null element `y` converted to type `T`.

# Examples
```julia-repl
julia> Vec1Dy{Float32}(2)
3-element Vec1Dy{Float32}:
 𝟎
 2.0f0
 𝟎

julia> Vec1Dy{ComplexF32}(Zero())
3-element Vec1Dy{ComplexF32}:
       𝟎
 0.0f0 + 0.0f0im
       𝟎

```
"""
@inline Vec1Dy{T}(a) where {T} = Vec(Zero(), convert(T, a), Zero())

"""
    Vec1Dy(y) -> Vec1Dy

Returns a 1D vector on the y direction with non-null element `y`.

# Examples
```julia-repl
julia> Vec1Dy(2.)
3-element Vec1Dy{Float64}:
 𝟎
 2.0
 𝟎

julia> Vec1Dy(One())
3-element Vec1Dy{One}:
 𝟎
 𝟏
 𝟎

```
"""
@inline Vec1Dy(a) = Vec1Dy{typeof(a)}(a)

"""
    Vec1Dz{T}(z) -> Vec1Dz{T}

Returns a 1D vector on the z direction with non-null element `z` converted to type `T`.

# Examples
```julia-repl
julia> Vec1Dz{Float32}(2)
3-element Vec1Dz{Float32}:
 𝟎
 𝟎
 2.0f0

julia> Vec1Dz{ComplexF32}(Zero())
3-element Vec1Dz{ComplexF32}:
       𝟎
       𝟎
 0.0f0 + 0.0f0im

```
"""
@inline Vec1Dz{T}(a) where {T} = Vec(Zero(), Zero(), convert(T, a))

"""
    Vec1Dz(z) -> Vec1Dz

Returns a 1D vector on the z direction with non-null element `z`.

# Examples
```julia-repl
julia> Vec1Dz(2.)
3-element Vec1Dz{Float64}:
 𝟎
 𝟎
 2.0

julia> Vec1Dz(One())
3-element Vec1Dz{One}:
 𝟎
 𝟎
 𝟏

```
"""
@inline Vec1Dz(a) = Vec1Dz{typeof(a)}(a)

@inline Vec0D() = Vec()

#Resolve ambiguities
@inline function Vec3D{Zero}(a,b)
    convert(Zero,a)
    convert(Zero,b)
    return Vec()
end

@inline function Vec3D{Zero}(a)
    convert(Zero,a)
    return Vec()
end

################# Vec constructors #################

################# Ten constructors #################

"""
    Ten(x::Vec, y::Vec, z::Vec) -> Tensor{2}

Returns a matrix (2nd order Tensor) whose 1st, 2nd and 3rd columns are given by the `x`, `y`, `z` vectors, respectively.
The elements of `x`, `y`, and `z` are promoted to a common type, with the exception of `Zero`s and `One`s.

# Examples
```julia-repl
julia> v1 = Vec(1,2,3)
3-element Vec3D{Int64}:
 1
 2
 3

julia> v2 = Vec(3.0, 2.0, One())
3-element Tensor{1, Union{One, Float64}, Float64, Float64, One}:
 3.0
 2.0
 𝟏

julia> v3 = Vec()
3-element Vec0D:
 𝟎
 𝟎
 𝟎

julia> T = Ten(v1, v2, v3)
3×3 Tensor{2, Union{One, Zero, Float64}, Vec3D{Float64}, Tensor{1, Union{One, Float64}, Float64, Float64, One}, Vec0D}:
 1.0  3.0  𝟎
 2.0  2.0  𝟎
 3.0  𝟏    𝟎
```
"""
@inline Ten(x::Vec, y::Vec, z::Vec) = Tensor(x,y,z)

"""
    Ten(xx, xy, xz,
        yx, yy, yz,
        zx, zy, zz) -> Tensor{2}

Returns a matrix (2nd order Tensor) equivalent to the literal Matrix `[xx xy xz; yx yy yz; zx zy zz]`.
All values are promoted to a common type, with the exception of `Zero`s and `One`s.

# Examples
```julia-repl
julia> T = Ten(1, 3.0, Zero(),
               2, 2.0, Zero(),
               3, One(), Zero())
3×3 Tensor{2, Union{One, Zero, Float64}, Vec3D{Float64}, Tensor{1, Union{One, Float64}, Float64, Float64, One}, Vec0D}:
 1.0  3.0  𝟎
 2.0  2.0  𝟎
 3.0  𝟏    𝟎

```
"""
@inline function Ten(xx, xy, xz,
                     yx, yy, yz,
                     zx, zy, zz)
    x = Vec(xx, yx, zx)
    y = Vec(xy, yy, zy)
    z = Vec(xz, yz, zz)
    return Ten(x, y, z)
end

"""
    Ten(;xx = Zero(), xy = Zero(), xz = Zero(),
         yx = Zero(), yy = Zero(), yz = Zero(),
         zx = Zero(), zy = Zero(), zz = Zero()) -> Tensor{2}

Returns a matrix (2nd order Tensor) equivalent to the literal Matrix `[xx xy xz; yx yy yz; zx zy zz]`.
All values are promoted to a common type, with the exception of `Zero`s and `One`s.

# Examples
```julia-repl
julia> T = Ten(xx = 1, xy = One(), yx = 2.0, zx = 3.0)
3×3 Tensor{2, Union{One, Zero, Float64}, Vec3D{Float64}, Vec1Dx{One}, Vec0D}:
 1.0  𝟏  𝟎
 2.0  𝟎  𝟎
 3.0  𝟎  𝟎

```
"""
@inline Ten(; xx = 𝟎, yx = 𝟎, zx = 𝟎, xy = 𝟎, yy = 𝟎, zy = 𝟎, xz = 𝟎, yz = 𝟎, zz = 𝟎) = 

    Ten(xx, xy, xz,
        yx, yy, yz,
        zx, zy, zz)


"""
    Ten3D{T}(x::Vec, y::Vec, z::Vec) -> Ten3D{T}

Returns a 3D matrix (2nd order Tensor) whose 1st, 2nd and 3rd columns are given by the `x`, `y`, `z` vectors, respectively.
The elements of `x`, `y`, and `z` are all converted to `T`.

# Examples
```julia-repl
julia> v1 = Vec(1,2,3)
3-element Vec3D{Int64}:
 1
 2
 3

julia> v2 = Vec(3.0, 2.0, One())
3-element Tensor{1, Union{One, Float64}, Float64, Float64, One}:
 3.0
 2.0
 𝟏

julia> v3 = Vec()
3-element Vec0D:
 𝟎
 𝟎
 𝟎

julia> T = Ten3D{Float32}(v1, v2, v3)
3×3 Ten3D{Float32}:
 1.0  3.0  0.0
 2.0  2.0  0.0
 3.0  1.0  0.0
```
"""
@inline Ten3D{T}(x::Vec, y::Vec, z::Vec) where {T} = Tensor(convert(Vec3D{T}, x), convert(Vec3D{T}, y), convert(Vec3D{T}, z))

"""
    Ten3D(x::Vec, y::Vec, z::Vec) -> Ten3D

Returns a 3D matrix (2nd order Tensor) whose 1st, 2nd and 3rd columns are given by the `x`, `y`, `z` vectors, respectively.
The elements of `x`, `y`, and `z` are all promoted to a common type.

# Examples
```julia-repl
julia> v1 = Vec(1,2,3)
3-element Vec3D{Int64}:
 1
 2
 3

julia> v2 = Vec(3.0, 2.0, One())
3-element Tensor{1, Union{One, Float64}, Float64, Float64, One}:
 3.0
 2.0
 𝟏

julia> v3 = Vec()
3-element Vec0D:
 𝟎
 𝟎
 𝟎

julia> T = Ten3D(v1, v2, v3)
3×3 Ten3D{Float64}:
 1.0  3.0  0.0
 2.0  2.0  0.0
 3.0  1.0  0.0
```
"""
@inline Ten3D(x::Vec, y::Vec, z::Vec) =
    Ten3D{promote_type(nonzero_eltype(x), nonzero_eltype(y), nonzero_eltype(z), Zero)}(x, y, z)

"""
    Ten3D{T}(xx, xy, xz,
             yx, yy, yz,
             zx, zy, zz) -> Ten3D{T}

Returns a matrix (2nd order Tensor) equivalent to the literal Matrix `[xx xy xz; yx yy yz; zx zy zz]` with all values converted to type `T`.

# Examples
```julia-repl
julia> T = Ten3D{Float32}(1, 3.0, Zero(),
                          2, 2.0, Zero(),
                          3, One(), Zero())
3×3 Ten3D{Float32}:
 1.0  3.0  0.0
 2.0  2.0  0.0
 3.0  1.0  0.0

```
"""
@inline Ten3D{T}(xx, xy, xz,
                 yx, yy, yz,
                 zx, zy, zz) where {T} = Ten3D{T}(Vec(xx, yx, zx), Vec(xy, yy, zy), Vec(xz, yz, zz))

"""
    Ten3D(xx, xy, xz,
          yx, yy, yz,
          zx, zy, zz) -> Ten3D

Returns a matrix (2nd order Tensor) equivalent to the literal Matrix `[xx xy xz; yx yy yz; zx zy zz]` with all values promoted to a common type.

# Examples
```julia-repl
julia> T = Ten3D(1, 3.0, Zero(),
                 2, 2.0, Zero(),
                 3, One(), Zero())
3×3 Ten3D{Float64}:
 1.0  3.0  0.0
 2.0  2.0  0.0
 3.0  1.0  0.0

```
"""
@inline Ten3D(xx, xy, xz,
              yx, yy, yz,
              zx, zy, zz) = 
    Ten3D{promote_type(typeof(xx), typeof(xy), typeof(xz),
                       typeof(yx), typeof(yy), typeof(yz),
                       typeof(zx), typeof(zy), typeof(zz))}(xx, xy, xz,
                                                            yx, yy, yz,
                                                            zx, zy, zz)

"""
    Ten2Dxy{T}(xx, xy, 
               yx, yy) -> Ten2Dxy{T}

Returns a 2D matrix (2nd order Tensor) on the x-y plane equivalent to the literal Matrix `[xx xy 𝟎; yx yy 𝟎; 𝟎 𝟎 𝟎]` with all input values converted to type `T`.

# Examples
```julia-repl
julia> T = Ten2Dxy{Float32}(One(), 2,
                            3, Zero())
3×3 Ten2Dxy{Float32}:
 1.0  2.0  𝟎
 3.0  0.0  𝟎
 𝟎    𝟎    𝟎

```
"""
@inline Ten2Dxy{T}(xx, xy, yx, yy) where {T} = 
    Ten(convert(T, xx), convert(T, xy), 𝟎,
        convert(T, yx), convert(T, yy), 𝟎,
        𝟎,              𝟎,              𝟎)

"""
    Ten2Dxy(xx, xy, 
            yx, yy) -> Ten2Dxy

Returns a 2D matrix (2nd order Tensor) on the x-y plane equivalent to the literal Matrix `[xx xy 𝟎; yx yy 𝟎; 𝟎 𝟎 𝟎]` with all input values promoted to a common type.

# Examples
```julia-repl
julia> T = Ten2Dxy(One(), 2,
                   3, Zero())
3×3 Ten2Dxy{Int64}:
 1  2  𝟎
 3  0  𝟎
 𝟎  𝟎  𝟎

```
"""
@inline Ten2Dxy(xx, xy, yx, yy) = 
    Ten2Dxy{promote_type(typeof(xx), typeof(xy),
                         typeof(yx), typeof(yy))}(xx, xy,
                                                  yx, yy)


"""
    Ten2Dxz{T}(xx, xz, 
               zx, zz) -> Ten2Dxz{T}

Returns a 2D matrix (2nd order Tensor) on the x-z plane equivalent to the literal Matrix `[xx 𝟎 xz; 𝟎 𝟎 𝟎; zx 𝟎 zz]` with all input values converted to type `T`.

# Examples
```julia-repl
julia> T = Ten2Dxz{Float32}(One(), 2,
                            3, Zero())
3×3 Ten2Dxz{Float32}:
 1.0  𝟎  2.0
 𝟎    𝟎  𝟎  
 3.0  𝟎  0.0

```
"""
@inline Ten2Dxz{T}(xx, xz, zx, zz) where {T} = 
    Ten(convert(T, xx), 𝟎, convert(T, xz),
        𝟎,              𝟎, 𝟎,
        convert(T, zx), 𝟎, convert(T, zz))

"""
    Ten2Dxz(xx, xz, 
            zx, zz) -> Ten2Dxz

Returns a 2D matrix (2nd order Tensor) on the x-z plane equivalent to the literal Matrix `[xx 𝟎 xz; 𝟎 𝟎 𝟎; zx 𝟎 zz]` with all input values promoted to a common type.

# Examples
```julia-repl
julia> T = Ten2Dxz(One(), 2,
                   3, Zero())
3×3 Ten2Dxz{Int64}:
 1  𝟎  2
 𝟎  𝟎  𝟎
 3  𝟎  0

```
"""
@inline Ten2Dxz(xx, xz, zx, zz) = 
    Ten2Dxz{promote_type(typeof(xx), typeof(xz),
                         typeof(zx), typeof(zz))}(xx, xz,
                                                  zx, zz)


"""
    Ten2Dyz{T}(yy, yz, 
               zy, zz) -> Ten2Dyz{T}

Returns a 2D matrix (2nd order Tensor) on the y-z plane equivalent to the literal Matrix `[𝟎 𝟎 𝟎; 𝟎 yy yz; 𝟎 zy zz]` with all input values converted to type `T`.

# Examples
```julia-repl
julia> T = Ten2Dyz{Float32}(One(), 2,
                            3, Zero())
3×3 Ten2Dyz{Float32}:
 𝟎  𝟎    𝟎
 𝟎  1.0  2.0
 𝟎  3.0  0.0

```
"""
@inline Ten2Dyz{T}(yy, yz, zy, zz) where {T} = 
    Ten(𝟎, 𝟎,              𝟎,
        𝟎, convert(T, yy), convert(T, yz),
        𝟎, convert(T, zy), convert(T, zz))

"""
    Ten2Dyz(yy, yz, 
            zy, zz) -> Ten2Dyz

Returns a 2D matrix (2nd order Tensor) on the y-z plane equivalent to the literal Matrix `[𝟎 𝟎 𝟎; 𝟎 yy yz; 𝟎 zy zz]` with all input values promoted to a common value.

# Examples
```julia-repl
julia> T = Ten2Dyz(One(), 2,
                   3, Zero())
3×3 Ten2Dyz{Int64}:
 𝟎  𝟎  𝟎
 𝟎  1  2
 𝟎  3  0

```
"""
@inline Ten2Dyz(yy, yz, zy, zz) = 
    Ten2Dyz{promote_type(typeof(yy), typeof(yz),
                         typeof(zy), typeof(zz))}(yy, yz,
                                                  zy, zz)


"""
    Ten1Dx{T}(xx) -> Ten1Dx{T}

Returns a 1D matrix (2nd order Tensor) on the x direction with non-null element `xx` converted to type `T`.

# Examples
```julia-repl
julia> Ten1Dx{Float32}(2)
3×3 Ten1Dx{Float32}:
 2.0  𝟎  𝟎
 𝟎    𝟎  𝟎
 𝟎    𝟎  𝟎

```
"""
@inline Ten1Dx{T}(xx) where {T} = Ten(convert(T, xx), 𝟎, 𝟎,
                                      𝟎,              𝟎, 𝟎,
                                      𝟎,              𝟎, 𝟎)

"""
    Ten1Dx(xx) -> Ten1Dx

Returns a 1D matrix (2nd order Tensor) on the x direction with non-null element `xx`.

# Examples
```julia-repl
julia> Ten1Dx(2)
3×3 Ten1Dx{Int64}:
 2  𝟎  𝟎
 𝟎  𝟎  𝟎
 𝟎  𝟎  𝟎

```
"""
@inline Ten1Dx(xx) = Ten1Dx{typeof(xx)}(xx)


"""
    Ten1Dy{T}(yy) -> Ten1Dy{T}

Returns a 1D matrix (2nd order Tensor) on the y direction with non-null element `yy` converted to type `T`.

# Examples
```julia-repl
julia> Ten1Dy{Float32}(2)
3×3 Ten1Dy{Float32}:
 𝟎  𝟎    𝟎
 𝟎  2.0  𝟎
 𝟎  𝟎    𝟎

```
"""
@inline Ten1Dy{T}(yy) where {T} = Ten(𝟎, 𝟎,              𝟎,
                                      𝟎, convert(T, yy), 𝟎,
                                      𝟎, 𝟎,              𝟎)

"""
    Ten1Dy(yy) -> Ten1Dy

Returns a 1D matrix (2nd order Tensor) on the y direction with non-null element `yy`.

# Examples
```julia-repl
julia> Ten1Dy(2)
3×3 Ten1Dy{Int64}:
 𝟎  𝟎  𝟎
 𝟎  2  𝟎
 𝟎  𝟎  𝟎

```
"""
@inline Ten1Dy(yy) = Ten1Dy{typeof(yy)}(yy)


"""
    Ten1Dz{T}(zz) -> Ten1Dz{T}

Returns a 1D matrix (2nd order Tensor) on the z direction with non-null element `zz` converted to type `T`.

# Examples
```julia-repl
julia> Ten1Dz{Float32}(2)
3×3 Ten1Dz{Float32}:
 𝟎  𝟎  𝟎
 𝟎  𝟎  𝟎
 𝟎  𝟎  2.0

```
"""
@inline Ten1Dz{T}(zz) where {T} = Ten(𝟎, 𝟎, 𝟎,
                                      𝟎, 𝟎, 𝟎,
                                      𝟎, 𝟎, convert(T, zz))

"""
    Ten1Dz(zz) -> Ten1Dz

Returns a 1D matrix (2nd order Tensor) on the z direction with non-null element `zz`.

# Examples
```julia-repl
julia> Ten1Dz(2)
3×3 Ten1Dz{Int64}:
 𝟎  𝟎  𝟎
 𝟎  𝟎  𝟎
 𝟎  𝟎  2

```
"""
@inline Ten1Dz(zz) = Ten1Dz{typeof(zz)}(zz)

@inline Ten0D() = Ten()

#Resolve ambiguities
@inline function Ten3D{Zero}(a,b,c,d)
    convert(Zero,a)
    convert(Zero,b)
    convert(Zero,c)
    convert(Zero,d)
    return Ten()
end

@inline function Ten3D{Zero}(a)
    convert(Zero,a)
    return Ten()
end

"""
    DiagTen(xx,yy,zz) -> DiagTen

Returns a diagonal matrix (2st order tensor) with `xx`, `yy`, `zz` as its diagonal elements.
The elements are promoted to a common type, with the exception of `Zero`s and `One`s, which maintains its type to denote any null or constant direction.

# Examples
```julia-repl
julia> T = TensorsLite.DiagTen(1,2.,One())
3×3 Tensor{2, Union{One, Zero, Float64}, Vec1Dx{Float64}, Vec1Dy{Float64}, Vec1Dz{One}}:
 1.0  𝟎    𝟎
 𝟎    2.0  𝟎
 𝟎    𝟎    𝟏

```
"""
@inline DiagTen(xx, yy, zz) = Ten(Vec1Dx(xx), Vec1Dy(yy), Vec1Dz(zz))

"""
    DiagTen3D{T}(xx, yy, zz) -> DiagTen3D{T}

Returns a diagonal matrix (2nd order Tensor) with diagonal elements `xx`, `yy`, `zz` converted to `T`.

# Examples
```julia-repl
julia> DiagTen3D{Float32}(One(), Zero(), 2)
3×3 DiagTen3D{Float32}:
 1.0  𝟎    𝟎
 𝟎    0.0  𝟎
 𝟎    𝟎    2.0

```
"""
@inline DiagTen3D{T}(xx, yy, zz) where {T} = Ten(Vec1Dx{T}(xx), Vec1Dy{T}(yy), Vec1Dz{T}(zz))

"""
    DiagTen3D(xx, yy, zz) -> DiagTen3D

Returns a diagonal matrix (2nd order Tensor) with diagonal elements `xx`, `yy`, `zz` promoted to a common type.

# Examples
```julia-repl
julia> DiagTen3D(One(), Zero(), 2)
3×3 DiagTen3D{Int64}:
 1  𝟎  𝟎
 𝟎  0  𝟎
 𝟎  𝟎  2

```
"""
@inline DiagTen3D(xx::Tx, yy::Ty, zz::Tz) where {Tx,Ty,Tz} = DiagTen3D{promote_type(Tx, Ty, Tz)}(xx, yy, zz)

"""
    DiagTen2Dxy{T}(xx, yy) -> DiagTen2Dxy{T}

Returns a 2D diagonal matrix (2nd order Tensor) on the x-y plane with diagonal elements `xx`, `yy` converted to `T`.

# Examples
```julia-repl
julia> DiagTen2Dxy{Float32}(One(), Zero())
3×3 DiagTen2Dxy{Float32}:
 1.0  𝟎    𝟎
 𝟎    0.0  𝟎
 𝟎    𝟎    𝟎

```
"""
@inline DiagTen2Dxy{T}(xx, yy) where {T} = Ten(Vec1Dx{T}(xx), Vec1Dy{T}(yy), Vec0D())

"""
    DiagTen2Dxy(xx, yy) -> DiagTen3D

Returns a 2D diagonal matrix (2nd order Tensor) on the x-y plane with diagonal elements `xx`, `yy` promoted to a common type.

# Examples
```julia-repl
julia> DiagTen2Dxy(One(), Zero())
3×3 DiagTen2Dxy{Bool}:
 true      𝟎  𝟎
    𝟎  false  𝟎
    𝟎      𝟎  𝟎

```
"""
@inline DiagTen2Dxy(xx::Tx, yy::Ty) where {Tx,Ty} = DiagTen2Dxy{promote_type(Tx, Ty)}(xx, yy)

"""
    DiagTen2Dxz{T}(xx, zz) -> DiagTen2Dxz{T}

Returns a 2D diagonal matrix (2nd order Tensor) on the x-z plane with diagonal elements `xx`, `zz` converted to `T`.

# Examples
```julia-repl
julia> DiagTen2Dxz{Float32}(One(), Zero())
3×3 DiagTen2Dxz{Float32}:
 1.0  𝟎    𝟎
 𝟎    𝟎    𝟎
 𝟎    𝟎    0.0

```
"""
@inline DiagTen2Dxz{T}(xx, zz) where {T} = Ten(Vec1Dx{T}(xx), Vec0D(), Vec1Dz{T}(zz))

"""
    DiagTen2Dxz(xx, zz) -> DiagTen2Dxz

Returns a 2D diagonal matrix (2nd order Tensor) on the x-z plane with diagonal elements `xx`, `zz` promoted to a common type.

# Examples
```julia-repl
julia> DiagTen2Dxz(One(), Zero())
3×3 DiagTen2Dxz{Bool}:
 true      𝟎  𝟎
    𝟎  false  𝟎
    𝟎      𝟎  𝟎

```
"""
@inline DiagTen2Dxz(xx::Tx, zz::Tz) where {Tx,Tz} = DiagTen2Dxz{promote_type(Tx, Tz)}(xx, zz)

"""
    DiagTen2Dyz{T}(yy, zz) -> DiagTen2Dyz{T}

Returns a 2D diagonal matrix (2nd order Tensor) on the y-z plane with diagonal elements `yy`, `zz` converted to `T`.

# Examples
```julia-repl
julia> DiagTen2Dyz{Float32}(One(), Zero())
3×3 DiagTen2Dyz{Float32}:
 𝟎    𝟎      𝟎
 𝟎    1.0    𝟎
 𝟎    𝟎      0.0

```
"""
@inline DiagTen2Dyz{T}(yy, zz) where {T} = Ten(Vec0D(), Vec1Dy{T}(yy), Vec1Dz{T}(zz))

"""
    DiagTen2Dyz(yy, zz) -> DiagTen2Dyz

Returns a 2D diagonal matrix (2nd order Tensor) on the y-z plane with diagonal elements `yy`, `zz` promoted to a common type.

# Examples
```julia-repl
julia> DiagTen2Dyz(One(), Zero())
3×3 DiagTen2Dyz{Bool}:
 𝟎     𝟎      𝟎
 𝟎  true      𝟎
 𝟎     𝟎  false

```
"""
@inline DiagTen2Dyz(yy::Ty, zz::Tz) where {Ty,Tz} = DiagTen2Dyz{promote_type(Ty, Tz)}(yy, zz)

#Ambiguities
@inline function DiagTen2Dxy{Zero}(xx, yy)
    convert(Zero, xx)
    convert(Zero, yy)
    return Ten()
end

################ Ten constructors #####################
