@inline function Ten(xx, xy, xz,
                     yx, yy, yz,
                     zx, zy, zz)
    x = Vec(xx, yx, zx)
    y = Vec(xy, yy, zy)
    z = Vec(xz, yz, zz)
    return Vec(x, y, z)
end

@inline Ten(; xx = 𝟎, yx = 𝟎, zx = 𝟎, xy = 𝟎, yy = 𝟎, zy = 𝟎, xz = 𝟎, yz = 𝟎, zz = 𝟎) = 
    Ten(xx, xy, xz,
        yx, yy, yz,
        zx, zy, zz)


const Ten3D{T} = Vec{T, 2, Vec3D{T}, Vec3D{T}, Vec3D{T}}

Ten3D{T}(xx, xy, xz, yx, yy, yz, zx, zy, zz) where {T} = 
    Ten(convert(T, xx), convert(T, xy), convert(T, xz),
        convert(T, yx), convert(T, yy), convert(T, yz),
        convert(T, zx), convert(T, zy), convert(T, zz))

Ten3D(xx, xy, xz, yx, yy, yz, zx, zy, zz) = 
    Ten3D{promote_type(typeof(xx), typeof(xy), typeof(xz),
                       typeof(yx), typeof(yy), typeof(yz),
                       typeof(zx), typeof(zy), typeof(zz))}(xx, xy, xz,
                                                            yx, yy, yz,
                                                            zx, zy, zz)


const Ten2Dxy{T} = Vec{Union{Zero, T}, 2, Vec2Dxy{T}, Vec2Dxy{T}, Vec0D}

Ten2Dxy{T}(xx, xy, yx, yy) where {T} = 
    Ten(convert(T, xx), convert(T, xy), 𝟎,
        convert(T, yx), convert(T, yy), 𝟎,
        𝟎,              𝟎,              𝟎)

Ten2Dxy(xx, xy, yx, yy) = 
    Ten2Dxy{promote_type(typeof(xx), typeof(xy),
                         typeof(yx), typeof(yy))}(xx, xy,
                                                  yx, yy)


const Ten2Dxz{T} = Vec{Union{Zero, T}, 2, Vec2Dxz{T}, Vec0D, Vec2Dxz{T}}

Ten2Dxz{T}(xx, xz, zx, zz) where {T} = 
    Ten(convert(T, xx), 𝟎, convert(T, xz),
        𝟎,              𝟎, 𝟎,
        convert(T, zx), 𝟎, convert(T, zz))

Ten2Dxz(xx, xz, zx, zz) = 
    Ten2Dxz{promote_type(typeof(xx), typeof(xz),
                         typeof(zx), typeof(zz))}(xx, xz,
                                                  zx, zz)


const Ten2Dyz{T} = Vec{Union{Zero, T}, 2, Vec0D, Vec2Dyz{T}, Vec2Dyz{T}}

Ten2Dyz{T}(yy, yz, zy, zz) where {T} = 
    Ten(𝟎, 𝟎,              𝟎,
        𝟎, convert(T, yy), convert(T, yz),
        𝟎, convert(T, zy), convert(T, zz))

Ten2Dyz(yy, yz, zy, zz) = 
Ten2Dyz{promote_type(typeof(yy), typeof(yz),
                     typeof(zy), typeof(zz))}(yy, yz,
                                              zy, zz)
 

const Ten2D{T} = Union{Ten2Dxy{T}, Ten2Dxz{T}, Ten2Dyz{T}}


const Ten1Dx{T} = Vec{Union{Zero, T}, 2, Vec1Dx{T}, Vec0D, Vec0D}

Ten1Dx{T}(xx) where {T} = Ten(convert(T, xx), 𝟎, 𝟎,
                              𝟎,              𝟎, 𝟎,
                              𝟎,              𝟎, 𝟎)

Ten1Dx(xx) = Ten1Dx{typeof(xx)}(xx)


const Ten1Dy{T} = Vec{Union{Zero, T}, 2, Vec0D, Vec1Dy{T}, Vec0D}

Ten1Dy{T}(yy) where {T} = Ten(𝟎, 𝟎,              𝟎,
                              𝟎, convert(T, yy), 𝟎,
                              𝟎, 𝟎,              𝟎)

Ten1Dy(yy) = Ten1Dy{typeof(yy)}(yy)


const Ten1Dz{T} = Vec{Union{Zero, T}, 2, Vec0D, Vec0D, Vec1Dz{T}}

Ten1Dz{T}(zz) where {T} = Ten(𝟎, 𝟎, 𝟎,
                              𝟎, 𝟎, 𝟎,
                              𝟎, 𝟎, convert(T, zz))

Ten1Dz(zz) = Ten1Dz{typeof(zz)}(zz)


const Ten1D{T} = Union{Ten1Dx{T}, Ten1Dy{T}, Ten1Dz{T}}
const TenND{T} = Union{Ten3D{T}, Ten2D{T}, Ten1D{T}}
const Ten0D = TenND{Zero}

const TenMaybe2Dxy{T, Tz} = Vec{Union{T, Tz}, 2, VecMaybe2Dxy{T, Tz}, VecMaybe2Dxy{T, Tz}, Vec3D{Tz}}

@inline Base.:*(T::AbstractTen, v::AbstractVec{<:Any, 1}) = dot(T, v)
@inline Base.:*(v::AbstractVec{<:Any, 1}, T::AbstractTen) = dot(v, T)

@inline function transpose(T::AbstractTen)
    x = T.x
    y = T.y
    z = T.z
    nx = Vec(x.x, y.x, z.x)
    ny = Vec(x.y, y.y, z.y)
    nz = Vec(x.z, y.z, z.z)
    return Vec(nx, ny, nz)
end

@inline function adjoint(T::AbstractTen)
    x = T.x
    y = T.y
    z = T.z
    nx = Vec(conj(x.x), conj(y.x), conj(z.x))
    ny = Vec(conj(x.y), conj(y.y), conj(z.y))
    nz = Vec(conj(x.z), conj(y.z), conj(z.z))
    return Vec(nx, ny, nz)
end

@inline dot(v::AbstractVec{<:Any, 1}, T::AbstractTen) = dot(transpose(T), v)

@inline function _muladd(T::AbstractTen, v::AbstractVec{<:Any, 1}, u::AbstractVec{<:Any, 1})
    Tt = transpose(T)
    return Vec(dotadd(Tt.x, v, u.x), dotadd(Tt.y, v, u.y), dotadd(Tt.z, v, u.z))
end
@inline muladd(T::AbstractTen, v::AbstractVec{<:Any, 1}, u::AbstractVec{<:Any, 1}) = _muladd(T, v, u)
@inline dotadd(T::AbstractTen, v::AbstractVec{<:Any, 1}, u::AbstractVec{<:Any, 1}) = _muladd(T, v, u)

@inline _muladd(v::AbstractVec{<:Any, 1}, T::AbstractTen, u::AbstractVec{<:Any, 1}) = Vec(dotadd(T.x, v, u.x), dotadd(T.y, v, u.y), dotadd(T.z, v, u.z))
@inline muladd(v::AbstractVec{<:Any, 1}, T::AbstractTen, u::AbstractVec{<:Any, 1}) = _muladd(v, T, u)
@inline dotadd(v::AbstractVec{<:Any, 1}, T::AbstractTen, u::AbstractVec{<:Any, 1}) = _muladd(v, T, u)

@inline Base.:*(T::AbstractTen, B::AbstractTen) = Vec(T * B.x, T * B.y, T * B.z)
@inline dot(T::AbstractTen, B::AbstractTen) = T * B

@inline _muladd(A::AbstractTen, B::AbstractTen, C::AbstractTen) = Vec(_muladd(A, B.x, C.x), _muladd(A, B.y, C.y), _muladd(A, B.z, C.z))
@inline muladd(A::AbstractTen, B::AbstractTen, C::AbstractTen) = _muladd(A, B, C)
@inline dotadd(A::AbstractTen, B::AbstractTen, C::AbstractTen) = _muladd(A, B, C)

@inline otimes(u::AbstractVec{<:Any, 1}, v::AbstractVec{<:Any, 1}) = Ten(
    u.x * v.x, u.x * v.y, u.x * v.z,
    u.y * v.x, u.y * v.y, u.y * v.z,
    u.z * v.x, u.z * v.y, u.z * v.z
)

const ⊗ = otimes

@inline inner(A::AbstractTen, B::AbstractTen) = dotadd(conj(A.x), B.x, dotadd(conj(A.y), B.y, dot(conj(A.z), B.z)))

@inline Base.sum(T::AbstractTen) = sum(T.x) + sum(T.y) + sum(T.z)
@inline Base.sum(op::F, T::AbstractTen) where {F <: Function} = sum(op, T.x) + sum(op, T.y) + sum(op, T.z)
