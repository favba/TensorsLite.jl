@inline fields(v::T) where T = ntuple(i->getfield(v,i),Val(fieldcount(T)))

@inline all_Numbers(a::T1,y::T2,z::T3) where {T1,T2,T3} = T1 <: Number && T2 <: Number && T3 <: Number

_my_promote_type(T::Type,Tx::Type) = (Tx === Zero || Tx === One) ? Tx : promote_type(T,Tx)

@inline function _final_type(types::Vararg{DataType,N})  where N
    _all_zeros(types...) && return Zero
    _all_zeros_and_ones(types...) && return Union{Zero,One}
    Tff = promote_type(types...)
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

@inline function _my_eltype(::Type{TA}) where TA<:AbstractArray
    T = eltype(TA)
    return _non_zero_type(T)
end
@inline _my_eltype(x::T) where T<:AbstractArray = _my_eltype(T)


@inline function _vec_type(Tx::Type,Ty::Type,Tz::Type)
    Tf = promote_type(Tx,Ty,Tz)
    Txf = _my_promote_type(Tf,Tx)
    Tyf = _my_promote_type(Tf,Ty)
    Tzf = _my_promote_type(Tf,Tz)
    Tff = _final_type(Txf,Tyf,Tzf)
    return Vec{Tff,1,Txf,Tyf,Tzf}
end

@inline _zero_for_tuple() = ()
@inline _zero_for_tuple(::Type{T},types::Vararg{T2,N}) where {T,T2,N} = (zero(T),_zero_for_tuple(types...)...)