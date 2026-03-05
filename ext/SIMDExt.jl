module SIMDExt

import TensorsLite: *, -, +
using TensorsLite, Zeros
import SIMD

@inline my_div(a,b) = a / b
@inline my_div(::Zero, ::SIMD.Vec) = Zero()
@inline Base.:/(v::T, b::SIMD.Vec) where {T <: AbstractTensor} = @inline  begin
    TensorsLite.constructor(T)(map(my_div, TensorsLite.fields(v), ntuple(i -> b, Val(fieldcount(T))))...)
end

@inline _getindex(::Type{Zero}, x, idx, rest::Vararg) = Zeros.Zero()
@inline _getindex(::Type{One}, x, idx, rest::Vararg) = Zeros.One()
Base.@propagate_inbounds _getindex(::Type, x, idx, rest::Vararg) = Base.getindex(x, idx, rest...)

const SIMDIndex{N} = Union{<:SIMD.VecRange{N}, <:SIMD.Vec{N, Int}} where {N}

Base.@propagate_inbounds function Base.getindex(arr::TensorArray{T, N, Tx, Ty, Tz}, idx::SIMDIndex, rest::Vararg) where {T, N, Tx, Ty, Tz}
    return @inline Tensor(
        _getindex(eltype(Tx), arr.x, idx, rest...),
        _getindex(eltype(Ty), arr.y, idx, rest...),
        _getindex(eltype(Tz), arr.z, idx, rest...),
    )
end

_convert(::Type{SIMD.Vec{N, T}}, ::Zero) where {N, T} = SIMD.Vec(ntuple(i -> zero(T), Val{N}())...)
_convert(::Type{SIMD.Vec{N, T}}, ::One) where {N, T} = SIMD.Vec(ntuple(i -> one(T), Val{N}())...)

@inline _setindex!(::Type{Zero}, x, v::Zero, idx::SIMDIndex, rest::Vararg) = v
@inline _setindex!(::Type{One}, x, v::One, idx::SIMDIndex, rest::Vararg) = v

# For When trying to write a higher-dimension Vec into a lower VecArray one
@inline function _setindex!(::Type{Zero}, x, v::SIMD.Vec{N, T}, idx, rest::Vararg) where {N, T}
    v == SIMD.Vec{N, T}(zero(T)) || throw(InexactError(:convert, Zero, v))
    return v
end

Base.@propagate_inbounds _setindex!(::Type{T}, x, v::Union{Zero, One}, idx::SIMDIndex{N}, rest::Vararg) where {T, N} = Base.setindex!(x, _convert(SIMD.Vec{N, T}, v), idx, rest...)
Base.@propagate_inbounds _setindex!(::Type, x, v, idx, rest::Vararg) = Base.setindex!(x, v, idx, rest...)

Base.@propagate_inbounds function Base.setindex!(arr::TensorArray{T, N, Tx, Ty, Tz}, v::AbstractTensor, idx::SIMDIndex, rest::Vararg) where {T, N, Tx, Ty, Tz}
    @inline begin
        _setindex!(eltype(Tx), arr.x, v.x, idx, rest...)
        _setindex!(eltype(Ty), arr.y, v.y, idx, rest...)
        _setindex!(eltype(Tz), arr.z, v.z, idx, rest...)
    end
    return v
end

Base.@propagate_inbounds function Base.getindex(
    arr::SymTenArray{T, N, Txx, Txy, Txz,
                                Tyy, Tyz,
                                     Tzz}, idx::SIMDIndex, rest::Vararg) where {T, N, Txx, Txy, Txz,
                                                                                           Tyy, Tyz,
                                                                                                Tzz}
    return @inline SymTen(
        _getindex(eltype(Txx), arr.xx, idx, rest...),
        _getindex(eltype(Txy), arr.xy, idx, rest...),
        _getindex(eltype(Txz), arr.xz, idx, rest...),
        _getindex(eltype(Tyy), arr.yy, idx, rest...),
        _getindex(eltype(Tyz), arr.yz, idx, rest...),
        _getindex(eltype(Tzz), arr.zz, idx, rest...)
    )
end

Base.@propagate_inbounds function Base.setindex!(
    arr::SymTenArray{T, N, Txx, Txy, Txz,
                                Tyy, Tyz,
                                     Tzz}, v::SymTen, idx::SIMDIndex, rest::Vararg) where {T, N, Txx, Txy, Txz,
                                                                                                      Tyy, Tyz,
                                                                                                           Tzz}

    @inline begin
        _setindex!(eltype(Txx), arr.xx, v.xx, idx, rest...)
        _setindex!(eltype(Txy), arr.xy, v.xy, idx, rest...)
        _setindex!(eltype(Txz), arr.xz, v.xz, idx, rest...)
        _setindex!(eltype(Tyy), arr.yy, v.yy, idx, rest...)
        _setindex!(eltype(Tyz), arr.yz, v.yz, idx, rest...)
        _setindex!(eltype(Tzz), arr.zz, v.zz, idx, rest...)
    end

    return v
end

Base.@propagate_inbounds function Base.getindex(arr::AntiSymTenArray{T, N, Txy, Txz, Tyz}, idx::SIMDIndex, rest::Vararg) where {T, N, Txy, Txz, Tyz}
    return @inline AntiSymTen(
        _getindex(eltype(Txy), arr.xy, idx, rest...),
        _getindex(eltype(Txz), arr.xz, idx, rest...),
        _getindex(eltype(Tyz), arr.yz, idx, rest...),
    )
end

Base.@propagate_inbounds function Base.setindex!(arr::AntiSymTenArray{T, N, Txy, Txz, Tyz}, v::AntiSymTen, idx::SIMDIndex, rest::Vararg) where {T, N, Txy, Txz, Tyz}
    @inline begin
        _setindex!(eltype(Txy), arr.xy, v.xy, idx, rest...)
        _setindex!(eltype(Txz), arr.xz, v.xz, idx, rest...)
        _setindex!(eltype(Tyz), arr.yz, v.yz, idx, rest...)
    end
    return v
end

#Is this type piracy?
Base.@propagate_inbounds function Base.setindex!(::AbstractArray{Zero, N}, v::Zero, ::SIMDIndex, ::Vararg) where {N}
    return v
end

Base.@propagate_inbounds function Base.setindex!(::AbstractArray{One, N}, v::One, ::SIMDIndex, ::Vararg) where {N}
    return v
end

end
