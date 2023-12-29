@inline function _final_type(Txf, Tyf, Tzf) 
    count(i->===(i,Zero),(Txf,Tyf,Tzf)) == 2 && count(i->===(i,One),(Txf,Tyf,Tzf)) == 1 && return Union{Zero,One}
    Tff = promote_type(Txf,Tyf,Tzf)
    (Txf !== Zero && Tyf !== Zero && Tzf !== Zero) ? Tff : Union{Zero,Tff}
end 

@inline all_Numbers(a::T1,y::T2,z::T3) where {T1,T2,T3} = T1 <: Number && T2 <: Number && T3 <: Number

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

@inline function _my_eltype(x::AbstractArray)
    T = eltype(x)
    return _non_zero_type(T)
end

