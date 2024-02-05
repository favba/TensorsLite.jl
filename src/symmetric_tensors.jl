struct SymTen{T,Txx,Tyx,Tzx,Tyy,Tzy,Tzz} <: AbstractVec{T,2}
    xx::Txx
    yx::Tyx
    zx::Tzx
    yy::Tyy
    zy::Tzy
    zz::Tzz

    @inline function SymTen(xx::Number,yx::Number,zx::Number,
                                       yy::Number,zy::Number,
                                                  zz::Number)
        Txx = typeof(xx)
        Tyx = typeof(yx)
        Tzx = typeof(zx)
        Tyy = typeof(yy)
        Tzy = typeof(zy)
        Tzz = typeof(zz)
        Tf = promote_type(Txx,Tyx,Tzx,
                              Tyy,Tzy,
                                  Tzz)
        xxn = _my_convert(Tf,xx)
        yxn = _my_convert(Tf,yx)
        zxn = _my_convert(Tf,zx)
        yyn = _my_convert(Tf,yy)
        zyn = _my_convert(Tf,zy)
        zzn = _my_convert(Tf,zz)

        Txxf = typeof(xxn)
        Tyxf = typeof(yxn)
        Tzxf = typeof(zxn)
        Tyyf = typeof(yyn)
        Tzyf = typeof(zyn)
        Tzzf = typeof(zzn)
        Tff = _final_type(Txxf,Tyxf,Tzxf,Tyyf,Tzyf,Tzzf)
        return new{Tff,Txxf,Tyxf,Tzxf,Tyyf,Tzyf,Tzzf}(xxn, yxn, zxn,
                                                           yyn, zyn,
                                                                zzn)
    end
end

@inline SymTen(;xx=𝟎, yx=𝟎, zx=𝟎, yy=𝟎, zy=𝟎, zz=𝟎) = SymTen(xx,yx,zx,yy,zy,zz)

@inline constructor(::Type{T}) where T<:SymTen = SymTen
@inline +(a::SymTen, b::SymTen) = @inline SymTen(map(+,fields(a),fields(b))...)
@inline -(a::SymTen, b::SymTen) = @inline SymTen(map(-,fields(a),fields(b))...)
@inline ==(a::SymTen,b::SymTen) = @inline reduce(&,map(==,fields(a),fields(b)))
@inline function _muladd(a::Number, v::SymTen, u::SymTen)
    @inline begin
        at = promote_type(typeof(a),_my_eltype(v),_my_eltype(u))(a)
        S = SymTen(map(_muladd,ntuple(i->at,Val(6)),fields(v),fields(u))...)
    end
    return S
end

const SymTen3D{T} = SymTen{T,T,T,T,T,T,T}
const SymTen2Dxy{T} = SymTen{Union{Zero,T},T,T,Zero,T,Zero,Zero}
const SymTen2Dxz{T} = SymTen{Union{Zero,T},T,Zero,T,Zero,Zero,T}
const SymTen2Dyz{T} = SymTen{Union{Zero,T},Zero,Zero,Zero,T,T,T}
const SymTen2D{T} = Union{SymTen2Dxy{T},SymTen2Dxz{T},SymTen2Dyz{T}}
const SymTen1Dx{T} = SymTen{Union{Zero,T},T,Zero,Zero,Zero,Zero,Zero}
const SymTen1Dy{T} = SymTen{Union{Zero,T},Zero,Zero,Zero,T,Zero,Zero}
const SymTen1Dz{T} = SymTen{Union{Zero,T},Zero,Zero,Zero,Zero,Zero,T}
const SymTen1D{T} = Union{SymTen1Dx{T},SymTen1Dy{T},SymTen1Dz{T}}

Base.IndexStyle(::Type{SymTen}) = IndexCartesian()
@inline function Base.getindex(S::SymTen,i::Integer,j::Integer)
    t = (Int(i),Int(j))
    @boundscheck checkbounds(S,t...)
    t === (1,1) && return S.xx
    (t === (2,1) || t === (1,2)) && return S.yx
    (t === (3,1) || t === (1,3)) && return S.zx
    t === (2,2) && return S.yy
    (t === (3,2) || t === (2,3))  && return S.zy
    return S.zz
end

@inline function Base.getproperty(S::SymTen,s::Symbol)
    if s === :x
        xx = getfield(S,:xx)
        yx = getfield(S,:yx)
        zx = getfield(S,:zx)
        return Vec(xx,yx,zx)
    elseif s === :y
        xy = getfield(S,:yx)
        yy = getfield(S,:yy)
        zy = getfield(S,:zy)
        return Vec(xy,yy,zy)
    elseif s === :z
        xz = getfield(S,:zx)
        yz = getfield(S,:zy)
        zz = getfield(S,:zz)
        return Vec(xz,yz,zz)
    else
        return getfield(S,s)
    end
end

@inline transpose(S::SymTen) = S
@inline adjoint(S::SymTen) = conj(transpose(S))
