import LinearAlgebra: transpose, adjoint

@inline function Ten(xx, yx, zx, xy, yy, zy, xz, yz, zz)
    x = Vec(xx, yx, zx)
    y = Vec(xy, yy, zy)
    z = Vec(xz, yz, zz)
    return Vec(x, y, z)
end
@inline Ten(; xx = 𝟎, yx = 𝟎, zx = 𝟎, xy = 𝟎, yy = 𝟎, zy = 𝟎, xz = 𝟎, yz = 𝟎, zz = 𝟎) = Ten(xx, yx, zx, xy, yy, zy, xz, yz, zz)

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
    xx = u.x * v.x, xy = u.x * v.y, xz = u.x * v.z,
    yx = u.y * v.x, yy = u.y * v.y, yz = u.y * v.z,
    zx = u.z * v.x, zy = u.z * v.y, zz = u.z * v.z
)

const ⊗ = otimes

@inline inner(A::AbstractTen, B::AbstractTen) = dotadd(conj(A.x), B.x, dotadd(conj(A.y), B.y, dot(conj(A.z), B.z)))

@inline dot(x::AbstractVec, A::AbstractTen, y::AbstractVec) = dot(dot(x, A), y)

@inline Base.sum(T::AbstractTen) = sum(T.x) + sum(T.y) + sum(T.z)
@inline Base.sum(op::F, T::AbstractTen) where {F <: Function} = sum(op, T.x) + sum(op, T.y) + sum(op, T.z)
