module SIMDExt

using TensorsLite, Zeros
import TensorsLite: *, -, +
import SIMD

TensorsLite.:*(::Zero, ::SIMD.Vec) = Zero()
TensorsLite.:*(::SIMD.Vec, ::Zero) = Zero()

TensorsLite.:+(::Zero, v::SIMD.Vec) = v
TensorsLite.:+(v::SIMD.Vec, ::Zero) = v

TensorsLite.:-(::Zero, v::SIMD.Vec) = -v
TensorsLite.:-(v::SIMD.Vec, ::Zero) = v

TensorsLite.:*(::One, v::SIMD.Vec) = v
TensorsLite.:*(v::SIMD.Vec, ::One) = v

TensorsLite.:+(::One, v::SIMD.Vec) = 1 + v
TensorsLite.:+(v::SIMD.Vec, ::One) = v + 1

TensorsLite.:-(::One, v::SIMD.Vec) = 1 - v
TensorsLite.:-(v::SIMD.Vec, ::One) = v - 1

@inline TensorsLite._muladd(::Zero, ::SIMD.Vec, ::Zero) = Zero()
@inline TensorsLite._muladd(::SIMD.Vec, ::Zero, ::Zero) = Zero()

@inline TensorsLite._muladd(::Zero, y::SIMD.Vec, x::SIMD.Vec) = convert(promote_type(typeof(y), typeof(x)), x)
@inline TensorsLite._muladd(::Zero, y::SIMD.Vec, x::Number) = Base.promote_op(+, typeof(y), typeof(x))(x)
@inline TensorsLite._muladd(::Zero, y::Number, x::SIMD.Vec) = Base.promote_op(+, typeof(y), typeof(x))(x)

@inline TensorsLite._muladd(y::SIMD.Vec, ::Zero, x::SIMD.Vec) = convert(promote_type(typeof(y), typeof(x)), x)
@inline TensorsLite._muladd(y::SIMD.Vec, ::Zero, x::Number) = Base.promote_op(+, typeof(y), typeof(x))(x)
@inline TensorsLite._muladd(y::Number, ::Zero, x::SIMD.Vec) = Base.promote_op(+, typeof(y), typeof(x))(x)

@inline TensorsLite._muladd(::Zero, ::Zero, x::SIMD.Vec) = x

@inline TensorsLite._muladd(x::SIMD.Vec, y::SIMD.Vec, ::Zero) = x * y
@inline TensorsLite._muladd(x::SIMD.Vec, y::Number, ::Zero) = x * y
@inline TensorsLite._muladd(x::Number, y::SIMD.Vec, ::Zero) = x * y

@inline TensorsLite._muladd(::One, x::SIMD.Vec, y::SIMD.Vec) = x + y
@inline TensorsLite._muladd(::One, x::SIMD.Vec, y::Number) = x + y
@inline TensorsLite._muladd(::One, x::Number, y::SIMD.Vec) = x + y

@inline TensorsLite._muladd(x::SIMD.Vec, ::One, y::SIMD.Vec) = x + y
@inline TensorsLite._muladd(x::SIMD.Vec, ::One, y::Number) = x + y
@inline TensorsLite._muladd(x::Number, ::One, y::SIMD.Vec) = x + y

@inline TensorsLite._muladd(x::SIMD.Vec{N, T}, y::SIMD.Vec{N, T}, ::One) where {N, T} = muladd(x, y, SIMD.Vec{N, T}(one(T)))
@inline TensorsLite._muladd(x::SIMD.Vec{N, T}, y::Number, ::One) where {N, T} = muladd(x, SIMD.Vec{N, T}(y), SIMD.Vec{N, T}(one(T)))
@inline TensorsLite._muladd(y::Number, x::SIMD.Vec{N, T}, ::One) where {N, T} = muladd(SIMD.Vec{N, T}(y), x, SIMD.Vec{N, T}(one(T)))

#Resolving Ambiguities
@inline TensorsLite._muladd(::One, ::Zero, x::SIMD.Vec) = x
@inline TensorsLite._muladd(::Zero, ::One, x::SIMD.Vec) = x

@inline TensorsLite._muladd(::One, x::SIMD.Vec, y::Zero) = x

@inline TensorsLite._muladd(x::SIMD.Vec, ::One, y::Zero) = x

@inline TensorsLite._muladd(::One, ::One, y::SIMD.Vec) = 1 + y

@inline TensorsLite._muladd(::One, x::SIMD.Vec, ::One) = x + 1

@inline TensorsLite._muladd(x::SIMD.Vec, ::One, ::One) = x + 1

@inline TensorsLite._muladd(::SIMD.Vec, ::Zero, ::One) = One()

@inline TensorsLite._muladd(::Zero, ::SIMD.Vec, ::One) = One()

@inline Base.:*(b::SIMD.Vec, v::T) where {T <: AbstractTensor} = @inline  begin
    TensorsLite.constructor(T)(map(TensorsLite.:*, ntuple(i -> b, Val(fieldcount(T))), TensorsLite.fields(v))...)
end

@inline my_div(a,b) = a / b
@inline my_div(::Zero, ::SIMD.Vec) = Zero()
@inline Base.:/(v::T, b::SIMD.Vec) where {T <: AbstractTensor} = @inline  begin
    TensorsLite.constructor(T)(map(my_div, TensorsLite.fields(v), ntuple(i -> b, Val(fieldcount(T))))...)
end

@inline Base.:*(v::AbstractTensor, b::SIMD.Vec) = b * v

@inline TensorsLite.dotadd(u::Vec, v::Vec, a::SIMD.Vec) = TensorsLite._muladd(u.x, v.x, TensorsLite._muladd(u.y, v.y, TensorsLite._muladd(u.z, v.z, a)))

@inline _getindex(::Type{Zero}, x, idx, rest::Vararg) = Zeros.Zero()
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
Base.@propagate_inbounds function Base.setindex!(a::AbstractArray{Zero, N}, v::Zero, idx::SIMDIndex, rest::Vararg) where {N}
    return v
end

end
