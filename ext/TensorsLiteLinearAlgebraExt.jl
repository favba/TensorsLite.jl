module TensorsLiteLinearAlgebraExt

using TensorsLite, Zeros

import TensorsLite: dot, Ten1D, SymTen1D, DiagTen, DiagSymTen

import LinearAlgebra: LinearAlgebra, norm, ⋅, cross, normalize

# Resolve method ambiguity for broadcasting, but I actually don't know when (or if) this
# case would ever happen and if this would be a proper fix
function Base.similar(bc::Base.Broadcast.Broadcasted{LinearAlgebra.StructuredMatrixStyle{T}}, ::Type{TT}) where {T, TT<:AbstractTensor}
    s = length.(axes(bc))
    return tensorarray(TT,s)
end

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
    return  Vec(muladd(ay, bz, -(az * by)), muladd(az, bx, -(ax * bz)), muladd(ax, by, -(ay * bx)))
end

base_type(::Type{T}) where {T} = T
base_type(::Type{Complex{T}}) where {T} = T
@inline Base.isapprox(x::AbstractTensor{N}, y::AbstractTensor{N}) where {N} = norm(x - y) <= max(Base.rtoldefault(base_type(nonzero_eltype(x))), Base.rtoldefault(base_type(nonzero_eltype(y)))) * max(norm(x), norm(y))

@inline LinearAlgebra.tr(T::Ten) = T.xx + T.yy + T.zz

@inline function LinearAlgebra.det(T::Ten)

    Txx = T.xx
    Tyx = T.yx
    Tzx = T.zx

    Txy = T.xy
    Tyy = T.yy
    Tzy = T.zy

    Txz = T.xz
    Tyz = T.yz
    Tzz = T.zz

    return muladd((Txy*Tyz - Txz*Tyy), Tzx, muladd((Tzy*Txz - Tzz*Txy), Tyx, (Tzz*Tyy - Tzy*Tyz )*Txx))

end

@inline LinearAlgebra.det(::AntiSymTen) = 𝟎

include("LinearAlgebra/eigen.jl")

end
