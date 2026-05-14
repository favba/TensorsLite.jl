module TensorsLiteSIMDExt

using Base: Constructor
import TensorsLite: dotadd, inneradd, dcontractadd, fields, constructor, sym_ten_fields
using TensorsLite, Zeros
import SIMD

@inline my_div(a,b) = a / b

@inline my_div(::Zero, ::SIMD.Vec) = Zero()

@inline Base.:/(v::T, b::SIMD.Vec) where {T <: AbstractTensor} = @inline  begin
    TensorsLite.constructor(T)(map(my_div, TensorsLite.fields(v), ntuple(i -> b, Val(fieldcount(T))))...)
end

@inline Base.:*(b::SIMD.Vec, v::T) where {T <: AbstractTensor} = @inline  begin
    constructor(T)(map(*, ntuple(i -> b, Val(fieldcount(T))), fields(v))...)
end

@inline Base.:*(T::AbstractTensor, b::SIMD.Vec) = b*T

@inline Base.muladd(a::SIMD.Vec, v::AbstractTensor{N}, u::AbstractTensor{N}) where {N} = Tensor(muladd(a, v.x, u.x), muladd(a, v.y, u.y), muladd(a, v.z, u.z))

@inline Base.muladd(b::SIMD.Vec, v::T, u::T) where {T <: AbstractTensor} = @inline  begin
    constructor(T)(map(muladd, ntuple(i -> b, Val(fieldcount(T))), fields(v), fields(u))...)
end

@inline Base.muladd(v::AbstractTensor{N}, a::SIMD.Vec, u::AbstractTensor{N}) where {N} = muladd(a, v, u)

@inline dotadd(a::Vec,b::Vec,c::SIMD.Vec) = muladd(a.x, b.x, muladd(a.y, b.y, muladd(a.z, b.z, c)))

@inline inneradd(u::Vec, v::Vec, c::SIMD.Vec) = dotadd(conj(u), v, c)

@inline inneradd(T1::AbstractTensor{N}, T2::AbstractTensor{N}, c::SIMD.Vec) where {N} = inneradd(T1.x, T2.x, inneradd(T1.y, T2.y, inneradd(T1.z, T2.z, c)))

@inline dcontractadd(a::Ten,b::Ten, c::SIMD.Vec) = muladd(a.xx, b.xx, muladd(a.xy, b.xy, muladd(a.xz, b.xz,
                                         muladd(a.yx, b.yx, muladd(a.yy, b.yy, muladd(a.yz, b.yz,
                                         muladd(a.zx, b.zx, muladd(a.zy, b.zy, muladd(a.zz, b.zz, c)))))))))

@inline function Base.muladd(a::SIMD.Vec, v::SymmetricTensor{2}, u::Union{<:TensorsLite.DiagTen,<:TensorsLite.Ten1D})
    @inline begin
        S = SymmetricTensor(map(muladd, ntuple(i -> a, Val(6)), sym_ten_fields(v), sym_ten_fields(u))...)
    end
    return S
end

@inline function Base.muladd(a::SIMD.Vec, v::Union{<:TensorsLite.DiagTen,<:TensorsLite.Ten1D}, u::SymmetricTensor{2})
    @inline begin
        S = SymmetricTensor(map(muladd, ntuple(i -> a, Val(6)), sym_ten_fields(v), sym_ten_fields(u))...)
    end
    return S
end

@inline inneradd(a::AntiSymmetricTensor{2, <:Union{Zero,SIMD.Vec}}, b::AntiSymmetricTensor{2, <:Union{Zero,SIMD.Vec}}, c::SIMD.Vec) = muladd(2, muladd(a.xy, b.xy, muladd(a.xz, b.xz, a.yz * b.yz)), c)

@inline inneradd(::AntiSymmetricTensor{2}, ::SymmetricTensor{2}, c::SIMD.Vec) = c

@inline inneradd(::SymmetricTensor{2}, ::AntiSymmetricTensor{2}, c::SIMD.Vec) = c


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

# Type piracy? Needed while https://github.com/eschnett/SIMD.jl/pull/157 isn't accepted.
Base.@propagate_inbounds Base.setindex!(a::AbstractArray{T}, s::Number, indx::SIMDIndex{N}, I::Vararg) where {T,N} = Base.setindex!(a, SIMD.Vec{N,T}(s), indx, I...)
Base.@propagate_inbounds Base.setindex!(a::AbstractArray{Zero}, s::Zero, indx::SIMDIndex{N}, I::Vararg) where {N} = a
Base.@propagate_inbounds Base.setindex!(a::AbstractArray{One}, s::One, indx::SIMDIndex{N}, I::Vararg) where {N} = a

end
