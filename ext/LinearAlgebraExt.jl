module LinearAlgebraExt

using TensorsLite, Zeros

import TensorsLite: dot, *, +, -, _muladd, transpose, adjoint

import LinearAlgebra: LinearAlgebra, norm

@inline LinearAlgebra.dot(a::AbstractVec, b::AbstractVec) = dot(a, b)
@inline LinearAlgebra.dot(x::AbstractVec, A::AbstractTen, y::AbstractVec) = dot(dot(x, A), y)

@inline fsqrt(x) = @fastmath sqrt(x)

@inline LinearAlgebra.norm(u::AbstractVec) = fsqrt(real(inner(u, u)))

@inline LinearAlgebra.norm(u::Vec1Dx, ::Real = 2) = abs(u.x)
@inline LinearAlgebra.norm(u::Vec1Dy, ::Real = 2) = abs(u.y)
@inline LinearAlgebra.norm(u::Vec1Dz, ::Real = 2) = abs(u.z)
@inline LinearAlgebra.norm(::TensorsLite.Vec0D, ::Real = 2) = 𝟎

@inline LinearAlgebra.normalize(u::AbstractVec) = u / norm(u)
@inline LinearAlgebra.normalize(u::AbstractVec{Zero}) = u

@inline function LinearAlgebra.cross(a::AbstractVec, b::AbstractVec)
    ax = a.x
    ay = a.y
    az = a.z
    bx = b.x
    by = b.y
    bz = b.z
    return  Vec(_muladd(ay, bz, -(az * by)), _muladd(az, bx, -(ax * bz)), _muladd(ax, by, -(ay * bx)))
end

@inline Base.isapprox(x::AbstractVec, y::AbstractVec) = norm(x - y) <= max(Base.rtoldefault(nonzero_eltype(x)), Base.rtoldefault(nonzero_eltype(y))) * max(norm(x), norm(y))

@inline LinearAlgebra.transpose(T::AbstractTen) = transpose(T)
@inline LinearAlgebra.adjoint(T::AbstractTen) = adjoint(T)

end
