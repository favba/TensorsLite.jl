@inline fields(v::T) where T = ntuple(i->getfield(v,i),Val(fieldcount(T)))

@inline no_Vecs(a::T1,y::T2,z::T3) where {T1,T2,T3} = !(T1 <: Vec) && !(T2 <: Vec) && !(T3 <: Vec)

_my_promote_type(T::Type,Tx::Type) = (Tx === Zero || Tx === One) ? Tx : promote_type(T,Tx)

@inline function _final_type(types::Vararg{DataType,N})  where N
    _all_zeros(types...) && return Zero
    _all_zeros_and_ones(types...) && return Union{Zero,One}
    Tff = promote_type_ignoring_Zero(types...)
    _is_there_any_zeros(types...) ? Union{Zero,Tff} : Tff
end 

@inline _all_zeros(types::Type{Zero}) = true
@inline _all_zeros(types::T) where T = false
@inline _all_zeros(::Type{Zero},types::Vararg{DataType,N}) where N = _all_zeros(types...)
@inline _all_zeros(::Type{T},types::Vararg{DataType,N}) where {T,N} =  false

@inline _all_zeros_and_ones(types::Type{Zero}) = true
@inline _all_zeros_and_ones(types::Type{One}) = true
@inline _all_zeros_and_ones(types::T) where T = false
@inline _all_zeros_and_ones(::Type{One},types::Vararg{DataType,N}) where N = _all_zeros_and_ones(types...)
@inline _all_zeros_and_ones(::Type{Zero},types::Vararg{DataType,N}) where N = _all_zeros_and_ones(types...)
@inline _all_zeros_and_ones(::DataType,types::Vararg{DataType,N}) where {N} =  false

@inline _is_there_any_zeros() = false
@inline _is_there_any_zeros(::Type{Zero},types::Vararg{DataType,N}) where N = true
@inline _is_there_any_zeros(::DataType,types::Vararg{DataType,N}) where {N} =  _is_there_any_zeros(types...)

function _my_convert(T::Type,x::T1) where T1
    T1 === Zero && return x
    T1 === One && return x
    return convert(T,x)
end

@inline function _non_zero_type(T::Type)
    if T isa Union
        T.a !== Zero && return T.a
        T.b !== Zero && return T.b
    else
        return T
    end
end

@inline function nonzero_eltype(::Type{TA}) where TA<:AbstractArray
    T = eltype(TA)
    return _non_zero_type(T)
end
@inline nonzero_eltype(x::T) where T<:AbstractArray = nonzero_eltype(T)

@inline _zero_for_tuple() = ()
@inline _zero_for_tuple(::Type{T},types::Vararg{T2,N}) where {T,T2,N} = (zero(T),_zero_for_tuple(types...)...)

@inline _filter_zeros() = ()
@inline _filter_zeros(::Zero,rest::Vararg{Any,N}) where N = (_filter_zeros(rest...)...,)
@inline _filter_zeros(x::Any,rest::Vararg{Any,N}) where N = (x,_filter_zeros(rest...)...,)

@inline _filter_type_zero() = ()
@inline _filter_type_zero(::Type{Zero}) = ()
@inline _filter_type_zero(t::Type) = (t,)
@inline _filter_type_zero(::Type{Zero},rest::Vararg{Type,N}) where N = (_filter_type_zero(rest...)...,)
@inline _filter_type_zero(x::Type,rest::Vararg{Type,N}) where N = (x,_filter_type_zero(rest...)...)

@inline _promote_type() = Zero
@inline _promote_type(types::Vararg{Any,N}) where N = promote_type(types...)
@inline promote_type_ignoring_Zero(types::Vararg{Any,N}) where N = _promote_type(_filter_type_zero(types...)...)