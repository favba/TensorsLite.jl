module TensorsLiteRandomExt

using TensorsLite, Random

import Random: SamplerType

Random.rand(rng::AbstractRNG, ::SamplerType{Tensor{N,T,Tx,Ty,Tz}}) where {N,T,Tx,Ty,Tz} =
    Tensor(rand(rng,Tx), rand(rng,Ty), rand(rng,Tz))

Random.rand(rng::AbstractRNG, ::SamplerType{SymmetricTensor{N,T,Txx,Txy,Txz,Tyy,Tyz,Tzz}}) where {N,T,Txx,Txy,Txz,Tyy,Tyz,Tzz} =
    SymmetricTensor(rand(rng,Txx), rand(rng,Txy), rand(rng,Txz), rand(rng,Tyy), rand(rng,Tyz), rand(rng,Tzz))

Random.rand(rng::AbstractRNG, ::SamplerType{AntiSymmetricTensor{N,T,Tx,Ty,Tz}}) where {N,T,Tx,Ty,Tz} =
    AntiSymmetricTensor(rand(rng,Tx), rand(rng,Ty), rand(rng,Tz))

Random.rand(::AbstractRNG, ::SamplerType{T}) where T<:Union{Zero,One} = T()

Random.rand(r::AbstractRNG, ::Type{T}, dims::Dims) where {T<:AbstractTensor} =
    rand!(r, tensorarray(T,dims))

end
