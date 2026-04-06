module TensorsLiteSIMDExt

using Base: Constructor
import TensorsLite: *, -, +, fields, constructor
using TensorsLite, Zeros
import SIMD

@inline my_div(a,b) = a / b
@inline my_div(::Zero, ::SIMD.Vec) = Zero()
@inline Base.:/(v::T, b::SIMD.Vec) where {T <: AbstractTensor} = @inline  begin
    TensorsLite.constructor(T)(map(my_div, TensorsLite.fields(v), ntuple(i -> b, Val(fieldcount(T))))...)
end

const SIMDIndex{N} = Union{<:SIMD.VecRange{N}, <:SIMD.Vec{N, Int}} where {N}

@inline _getindex(x::AbstractArray{Zero}, idx::SIMDIndex, rest::Vararg) = Zeros.Zero()
@inline _getindex(x::AbstractArray{One}, idx::SIMDIndex, rest::Vararg) = Zeros.One()
Base.@propagate_inbounds _getindex(x::AbstractArray, idx::SIMDIndex, rest::Vararg) = Base.getindex(x, idx, rest...)

struct SIndexHelper{T<:SIMDIndex,N}
    idx::T
    i::NTuple{N,Int}
end

@inline (s::SIndexHelper{T,N})(A) where {T,N} = @inbounds(_getindex(A, s.idx, (s.i)...))

Base.@inline function Base.getindex(arr::AbstractTensorArray{T, N}, idx::SIMDIndex, rest::Vararg) where {T, N}
    @boundscheck checkbounds(arr, idx, rest...)
    r = constructor(T)(map(SIndexHelper(idx,rest), fields(arr))...)
    return r
end

_convert(::Type{SIMD.Vec{N, T}}, ::Zero) where {N, T} = SIMD.Vec(ntuple(i -> zero(T), Val{N}())...)
_convert(::Type{SIMD.Vec{N, T}}, ::One) where {N, T} = SIMD.Vec(ntuple(i -> one(T), Val{N}())...)

@inline _setindex!(x::AbstractArray{Zero}, v::Zero, idx::SIMDIndex, rest::Vararg) = v
@inline _setindex!(x::AbstractArray{One}, v::One, idx::SIMDIndex, rest::Vararg) = v

# For When trying to write a higher-dimension Vec into a lower VecArray one
@inline function _setindex!(x::AbstractArray{Zero}, v::SIMD.Vec{N, T}, idx, rest::Vararg) where {N, T}
    v == SIMD.Vec{N, T}(zero(T)) || throw(InexactError(:convert, Zero, v))
    return v
end

Base.@propagate_inbounds _setindex!(x::AbstractArray{T}, v::Union{Zero, One}, idx::SIMDIndex{N}, rest::Vararg) where {T, N} = Base.setindex!(x, _convert(SIMD.Vec{N, T}, v), idx, rest...)
Base.@propagate_inbounds _setindex!(x::AbstractArray, v, idx::SIMDIndex, rest::Vararg) = Base.setindex!(x, v, idx, rest...)

@inline (s::SIndexHelper{T,N})(A,v) where {T,N} = @inbounds(_setindex!(A,v,s.idx, (s.i)...))

Base.@propagate_inbounds function Base.setindex!(arr::AbstractTensorArray{T, N}, v::AbstractTensor, idx::SIMDIndex, rest::Vararg) where {T, N}
    @boundscheck checkbounds(arr, idx, rest...)
    @inline map(SIndexHelper(idx,rest), fields(arr), fields(v))
    return arr
end

#Is this type piracy?
# Base.@propagate_inbounds function Base.setindex!(::AbstractArray{Zero, N}, v::Zero, ::SIMDIndex, ::Vararg) where {N}
#     return v
# end

# Base.@propagate_inbounds function Base.setindex!(::AbstractArray{One, N}, v::One, ::SIMDIndex, ::Vararg) where {N}
#     return v
# end

# Type piracy? Needed while https://github.com/eschnett/SIMD.jl/pull/157 isn't accepted.
Base.@propagate_inbounds Base.setindex!(a::AbstractArray{T}, s::Number, indx::SIMDIndex{N}, I::Vararg) where {T,N} = Base.setindex!(a, SIMD.Vec{N,T}(s), indx, I...)
Base.@propagate_inbounds Base.setindex!(a::AbstractArray{Zero}, s::Zero, indx::SIMDIndex{N}, I::Vararg) where {N} = a
Base.@propagate_inbounds Base.setindex!(a::AbstractArray{One}, s::One, indx::SIMDIndex{N}, I::Vararg) where {N} = a

end
