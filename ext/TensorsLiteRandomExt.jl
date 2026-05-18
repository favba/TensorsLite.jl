module TensorsLiteRandomExt

using TensorsLite, Random

import Random: SamplerType

Random.rand(rng::AbstractRNG, ::SamplerType{Tensor{N,T,Tx,Ty,Tz}}) where {N,T,Tx,Ty,Tz} =
    Tensor(rand(rng,Tx), rand(rng,Ty), rand(rng,Tz))

Random.rand(rng::AbstractRNG, ::SamplerType{SymmetricTensor{N,T,Txx,Txy,Txz,Tyy,Tyz,Tzz}}) where {N,T,Txx,Txy,Txz,Tyy,Tyz,Tzz} =
    SymmetricTensor(rand(rng,Txx), rand(rng,Txy), rand(rng,Txz), rand(rng,Tyy), rand(rng,Tyz), rand(rng,Tzz))

Random.rand(rng::AbstractRNG, ::SamplerType{AntiSymmetricTensor{N,T,Tx,Ty,Tz}}) where {N,T,Tx,Ty,Tz} =
    AntiSymmetricTensor(rand(rng,Tx), rand(rng,Ty), rand(rng,Tz))

Random.rand(r::AbstractRNG, ::Type{T}, dims::Dims) where {T<:AbstractTensor} =
    rand!(r, tensorarray(T,dims))

# Make things like `rand(Vec3D)` default to `rand(Vec3D{Float64})`
for TT in (:Vec3D, :Vec2Dxy, :Vec2Dxz, :Vec2Dyz, :Vec1Dx, :Vec1Dy, :Vec1Dz,
    :Ten3D, :Ten2Dxy, :Ten2Dxz, :Ten2Dyz, :Ten1Dx, :Ten1Dy, :Ten1Dz,
    :DiagTen3D, :DiagTen2Dxy, :DiagTen2Dxz, :DiagTen2Dyz,
    :SymTen3D, :SymTen2Dxy, :SymTen2Dxz, :SymTen2Dyz, :SymTen1Dx, :SymTen1Dy, :SymTen1Dz,
    :DiagSymTen3D, :DiagSymTen2Dxy, :DiagSymTen2Dxz, :DiagSymTen2Dyz,
    :AntiSymTen3D, :AntiSymTen2Dxy, :AntiSymTen2Dxz, :AntiSymTen2Dyz,
    )

    @eval Random.rand(rng::AbstractRNG, ::SamplerType{$(TT)}) = Random.rand(rng, $(Expr(:curly, TT, :Float64)))

    @eval Random.rand(r::AbstractRNG, ::Type{$(TT)}, dims::Dims) = rand!(r, tensorarray($(Expr(:curly, TT, :Float64)),dims))
end

end
