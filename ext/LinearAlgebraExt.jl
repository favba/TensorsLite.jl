module LinearAlgebraExt

using TensorsLite, Zeros

import TensorsLite: dot, *, +, -, _muladd

import LinearAlgebra: LinearAlgebra, norm

@inline LinearAlgebra.dot(a::AbstractTensor, b::AbstractTensor) = dot(a, b)
@inline LinearAlgebra.dot(x::Vec, A::Ten, y::Vec) = dot(dot(x, A), y)

@inline fsqrt(x) = @fastmath sqrt(x)

@inline LinearAlgebra.norm(u::AbstractTensor) = fsqrt(real(inner(u, u)))

@inline LinearAlgebra.norm(u::Vec1Dx, ::Real = 2) = abs(u.x)
@inline LinearAlgebra.norm(u::Vec1Dy, ::Real = 2) = abs(u.y)
@inline LinearAlgebra.norm(u::Vec1Dz, ::Real = 2) = abs(u.z)
@inline LinearAlgebra.norm(::VecND{Zero}, ::Real = 2) = 𝟎

@inline LinearAlgebra.normalize(u::AbstractTensor) = u / norm(u)
@inline LinearAlgebra.normalize(u::AbstractTensor{N,Zero}) where {N} = u

@inline function LinearAlgebra.cross(a::Vec, b::Vec)
    ax = a.x
    ay = a.y
    az = a.z
    bx = b.x
    by = b.y
    bz = b.z
    return  Vec(_muladd(ay, bz, -(az * by)), _muladd(az, bx, -(ax * bz)), _muladd(ax, by, -(ay * bx)))
end

@inline Base.isapprox(x::AbstractTensor{N}, y::AbstractTensor{N}) where {N} = norm(x - y) <= max(Base.rtoldefault(nonzero_eltype(x)), Base.rtoldefault(nonzero_eltype(y))) * max(norm(x), norm(y))

@inline LinearAlgebra.tr(T::Ten) = T.x.x + T.y.y + T.z.z

@inline function LinearAlgebra.det(T::Ten)

    Tx = T.x
    Txx = Tx.x
    Tyx = Tx.y
    Tzx = Tx.z

    Ty = T.y
    Txy = Ty.x
    Tyy = Ty.y
    Tzy = Ty.z

    Tz = T.z
    Txz = Tz.x
    Tyz = Tz.y
    Tzz = Tz.z

    return _muladd((Txy*Tyz - Txz*Tyy), Tzx, _muladd((Tzy*Txz - Tzz*Txy), Tyx, (Tzz*Tyy - Tzy*Tyz )*Txx))

end

@inline LinearAlgebra.det(::AntiSymTen) = 𝟎

end
