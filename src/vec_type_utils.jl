@inline _eltype_convert(::Type{T}, number) where {T} = _my_convert(T, number)

@inline function _eltype_convert(::Type{T}, vec::AbstractVec{Tv, N}) where {T, Tv, N}
    Vec(_eltype_convert(T, vec.x), _eltype_convert(T, vec.y), _eltype_convert(T, vec.z))
end
