module SIMDExt

using TensorsLite, Zeros
import SIMD

Base.conj(x::SIMD.Vec{N, T}) where {N, T} = x
Base.real(x::SIMD.Vec{N, T}) where {N, T} = x

Base.convert(::Type{SIMD.Vec{N, T}}, ::Zero) where {N, T} = SIMD.Vec(ntuple(i -> zero(T), Val{N}())...)
Base.isapprox(::Zero, v::SIMD.Vec{N, T}) where {N, T} = SIMD.Vec(ntuple(i -> zero(T), Val{N}())...) == v
Base.isapprox(v::SIMD.Vec{N, T}, ::Zero) where {N, T} = SIMD.Vec(ntuple(i -> zero(T), Val{N}())...) == v

Base.:*(::Zero, ::SIMD.Vec) = Zero()
Base.:*(::SIMD.Vec, ::Zero) = Zero()

Base.:+(::Zero, v::SIMD.Vec) = v
Base.:+(v::SIMD.Vec, ::Zero) = v

Base.:-(::Zero, v::SIMD.Vec) = -v
Base.:-(v::SIMD.Vec, ::Zero) = v

Base.:*(::One, v::SIMD.Vec) = v
Base.:*(v::SIMD.Vec, ::One) = v

Base.:+(::One, v::SIMD.Vec) = 1 + v
Base.:+(v::SIMD.Vec, ::One) = v + 1

Base.:-(::One, v::SIMD.Vec) = 1 - v
Base.:-(v::SIMD.Vec, ::One) = v - 1

@inline TensorsLite._muladd(::Zero, ::SIMD.Vec, ::Zero) = Zero()
@inline TensorsLite._muladd(::SIMD.Vec, ::Zero, ::Zero) = Zero()
@inline TensorsLite._muladd(::Zero, y::SIMD.Vec, x::SIMD.Vec) = convert(promote_type(typeof(y), typeof(x)), x)
@inline TensorsLite._muladd(y::SIMD.Vec, ::Zero, x::SIMD.Vec) = convert(promote_type(typeof(y), typeof(x)), x)
@inline TensorsLite._muladd(::Zero, ::Zero, x::SIMD.Vec) = x
@inline TensorsLite._muladd(x::SIMD.Vec, y::SIMD.Vec, ::Zero) = x * y
@inline TensorsLite._muladd(::One, x::SIMD.Vec, y::SIMD.Vec) = x + y
@inline TensorsLite._muladd(x::SIMD.Vec, ::One, y::SIMD.Vec) = x + y

#Resolving Ambiguities
@inline TensorsLite._muladd(::One, ::Zero, x::SIMD.Vec) = x
@inline TensorsLite._muladd(::Zero, ::One, x::SIMD.Vec) = x

@inline TensorsLite._muladd(::One, x::SIMD.Vec, y::Zero) = x

@inline TensorsLite._muladd(x::SIMD.Vec, ::One, y::Zero) = x

@inline TensorsLite._muladd(::One, ::One, y::SIMD.Vec) = One() + y

@inline Base.:/(v::AbstractVec, b::SIMD.Vec) = inv(b) * v

@inline Base.:*(b::SIMD.Vec, v::T) where {T <: AbstractVec} = @inline  begin
    bt = convert(promote_type(typeof(b), TensorsLite.nonzero_eltype(T)), b)
    TensorsLite.constructor(T)(map(*, ntuple(i -> bt, Val(fieldcount(T))), TensorsLite.fields(v))...)
end

@inline Base.:*(v::AbstractVec, b::SIMD.Vec) = b * v


@inline TensorsLite.dotadd(u::AbstractVec, v::AbstractVec, a::SIMD.Vec) = TensorsLite._muladd(u.x, v.x, TensorsLite._muladd(u.y, v.y, TensorsLite._muladd(u.z, v.z, a)))

end
