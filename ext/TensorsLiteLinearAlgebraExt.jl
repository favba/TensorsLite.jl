module TensorsLiteLinearAlgebraExt

using TensorsLite, Zeros

import TensorsLite: dot, DiagTen, DiagSymTen, DDiagTen, DDiagSymTen
import TensorsLite: QuasiTen2Dxy, QuasiTen2Dxz, QuasiTen2Dyz
import TensorsLite: QuasiSymTen2Dxy, QuasiSymTen2Dxz, QuasiSymTen2Dyz
import TensorsLite: DUTriTen, DLTriTen

import LinearAlgebra: LinearAlgebra, norm, ⋅, cross, normalize, ×

# Resolve method ambiguity for broadcasting, but I actually don't know when (or if) this
# case would ever happen and if this would be a proper fix
function Base.similar(bc::Base.Broadcast.Broadcasted{LinearAlgebra.StructuredMatrixStyle{T}}, ::Type{TT}) where {T, TT<:AbstractTensor}
    s = length.(axes(bc))
    return tensorarray(TT,s)
end

"""
    dot(a::AbstractTensor{N}, b::AbstractTensor{M}) -> AbstractTensor{N + M - 2}

Contracts (sums over the product of the elements) the innermost indices of tensors `a` and `b`. The result is a tensor of order `N + M - 2`. If `N == M == 1` returns a `Number`.
See also [`dotadd`](@ref), [`tdot`](@ref), [`dott`](@ref) and [`inner`](@ref).
"""
@inline LinearAlgebra.dot(a::AbstractTensor, b::AbstractTensor) = dot(a, b)

"""
    dot(a::AbstractTensor{N}) -> AbstractTensor{2N - 2}

Equivalent to `dot(a, a)`. Returns a `SymTen` if `a` is either a `SymTen` or `AntiSymTen`.
"""
@inline LinearAlgebra.dot(a::AbstractTensor) = dot(a)

@inline LinearAlgebra.dot(x::Vec, A::Ten, y::Vec) = dot(dot(x, A), y)

@inline fsqrt(x) = @fastmath sqrt(x)

@inline LinearAlgebra.norm(u::AbstractTensor) = fsqrt(real(inner(u, u)))

@inline LinearAlgebra.norm(u::Vec1Dx, ::Real = 2) = abs(u.x)

@inline LinearAlgebra.norm(u::Vec1Dy, ::Real = 2) = abs(u.y)

@inline LinearAlgebra.norm(u::Vec1Dz, ::Real = 2) = abs(u.z)

@inline LinearAlgebra.norm(::Vec0D, ::Real = 2) = 𝟎

@inline LinearAlgebra.normalize(u::AbstractTensor) = u / norm(u)

@inline LinearAlgebra.normalize(u::AbstractTensor{N,Zero}) where {N} = u

@inline LinearAlgebra.cross(a::Vec, b::Vec) = TensorsLite.cross(a, b)

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


include("LinearAlgebra/lu.jl")
include("LinearAlgebra/qr.jl")
include("LinearAlgebra/eigen.jl")

# include("LinearAlgebra/inv.jl")

end
