@inline fields(v::T) where {T} = ntuple(i -> getfield(v, i), Val(fieldcount(T)))

_my_promote_type(T::Type, Tx::Type) = (Tx === Zero || Tx === One) ? Tx : promote_type(T, Tx)

function _my_convert(T::Type, x::T1) where {T1}
    T1 === Zero && return x
    T1 === One && return x
    return convert(T, x)
end

@inline function __non_StaticBool_type(T::Type)
    if T isa Union
        if T.b isa Union
            return _filter_type_zero_and_one(T.a, T.b.a, T.b.b)[1]
        else
            r = _filter_type_zero_and_one(T.a, T.b)
            return isempty(r) ? One : r[1]
        end
    else
        return T
    end
end

"""
    _non_StaticBool_type(::Type{Union{T,Zero,One}) -> T
    _non_StaticBool_type(::Type{Union{T,Zero}) -> T
    _non_StaticBool_type(::Type{Union{T,One}) -> T
    _non_StaticBool_type(::Type{Union{Zero,One}) -> One
"""
_non_StaticBool_type(::Type{T}) where {T} = __non_StaticBool_type(T)

"""
    nonzero_eltype(::Type{AbstractArray{Union{T,Zero}}) -> T
    nonzero_eltype(::Type{AbstractArray{Union{T,Zero,One}}) -> T
    nonzero_eltype(::Type{AbstractArray{Union{T,One}}) -> T
    nonzero_eltype(::Type{AbstractArray{Union{Zero,One}}) -> One
"""
@inline function nonzero_eltype(::Type{TA}) where {TA <: AbstractArray}
    T = eltype(TA)
    return _non_StaticBool_type(T)
end
@inline nonzero_eltype(::T) where {T <: AbstractArray} = nonzero_eltype(T)

@inline _zero_for_tuple() = ()
@inline _zero_for_tuple(::Type{T}, types::Vararg{T2, N}) where {T, T2, N} = (zero(T), _zero_for_tuple(types...)...)

@inline _filter_zeros() = ()
@inline _filter_zeros(::Zero, rest::Vararg{Any, N}) where {N} = (_filter_zeros(rest...)...,)
@inline _filter_zeros(x::Any, rest::Vararg{Any, N}) where {N} = (x, _filter_zeros(rest...)...)

@inline _filter_type_zero_and_one() = ()
@inline _filter_type_zero_and_one(::Type{Zero}) = ()
@inline _filter_type_zero_and_one(::Type{One}) = ()
@inline _filter_type_zero_and_one(t::Type) = (t,)
@inline _filter_type_zero_and_one(::Type{Zero}, rest::Vararg{Type, N}) where {N} = (_filter_type_zero_and_one(rest...)...,)
@inline _filter_type_zero_and_one(::Type{One}, rest::Vararg{Type, N}) where {N} = (_filter_type_zero_and_one(rest...)...,)
@inline _filter_type_zero_and_one(x::Type, rest::Vararg{Type, N}) where {N} = (x, _filter_type_zero_and_one(rest...)...)

@inline _promote_type() = Zero
@inline _promote_type(types::Vararg{Any, N}) where {N} = promote_type(types...)
@inline promote_type_ignoring_Zero_and_One(types::Vararg{Any, N}) where {N} = _promote_type(_filter_type_zero_and_one(types...)...)
