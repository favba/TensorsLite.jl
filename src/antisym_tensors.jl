struct AntiSymTen{T,Tyx,Tzx,Tzy} <: AbstractVec{T,2}
    yx::Tyx
    zx::Tzx
    zy::Tzy

    @inline function AntiSymTen(yx::Number,zx::Number,zy::Number)
        Tyx = typeof(yx)
        Tzx = typeof(zx)
        Tzy = typeof(zy)
        Tf = promote_type(Tyx,Tzx,Tzy)
        yxn = _my_convert(Tf,yx)
        zxn = _my_convert(Tf,zx)
        zyn = _my_convert(Tf,zy)

        Tyxf = typeof(yxn)
        Tzxf = typeof(zxn)
        Tzyf = typeof(zyn)
        Tff = _final_type(Tyxf,Tzxf,Tzyf)
        return new{Tff,Tyxf,Tzxf,Tzyf}(yxn, zxn,zyn)
    end
end

@inline AntiSymTen(;yx=𝟎, zx=𝟎, zy=𝟎) = AntiSymTen(yx,zx,zy)

@inline constructor(::Type{T}) where T<:AntiSymTen = AntiSymTen
@inline +(a::AntiSymTen, b::AntiSymTen) = @inline AntiSymTen(map(+,fields(a),fields(b))...)
@inline -(a::AntiSymTen, b::AntiSymTen) = @inline AntiSymTen(map(-,fields(a),fields(b))...)
@inline ==(a::AntiSymTen,b::AntiSymTen) = @inline reduce(&,map(==,fields(a),fields(b)))
@inline function _muladd(a::Number, v::AntiSymTen, u::AntiSymTen)
    @inline begin
        at = promote_type(typeof(a),_my_eltype(v),_my_eltype(u))(a)
        W = AntiSymTen(map(_muladd,ntuple(i->at,Val(3)),fields(v),fields(u))...)
    end
    return W
end


const AntiSymTen3D{T} = AntiSymTen{T,T,T,T}
const AntiSymTen2Dxy{T} = AntiSymTen{Union{Zero,T},T,Zero,Zero}
const AntiSymTen2Dxz{T} = AntiSymTen{Union{Zero,T},Zero,T,Zero}
const AntiSymTen2Dyz{T} = AntiSymTen{Union{Zero,T},Zero,Zero,T}

Base.IndexStyle(::Type{AntiSymTen}) = IndexCartesian()
@inline function Base.getindex(S::AntiSymTen,i::Integer,j::Integer)
    t = (Int(i),Int(j))
    @boundscheck checkbounds(S,t...)
    t === (1,1) && return 𝟎
    t === (2,1) && return S.yx
    t === (1,2) && return -S.yx
    t === (3,1) && return S.zx
    t === (1,3) && return -S.zx
    t === (2,2) && return 𝟎
    t === (3,2) && return S.zy
    t === (2,3) && return -S.zy
    return 𝟎
end

@inline function Base.getproperty(S::AntiSymTen,s::Symbol)
    if s === :x
        xx = 𝟎
        yx = getfield(S,:yx)
        zx = getfield(S,:zx)
        return Vec(xx,yx,zx)
    elseif s === :y
        xy = -getfield(S,:yx)
        yy = 𝟎
        zy = getfield(S,:zy)
        return Vec(xy,yy,zy)
    elseif s === :z
        xz = -getfield(S,:zx)
        yz = -getfield(S,:zy)
        zz = 𝟎
        return Vec(xz,yz,zz)
    else
        return getfield(S,s)
    end
end

@inline transpose(W::AntiSymTen) = -W
@inline adjoint(W::AntiSymTen) = -conj(W)
