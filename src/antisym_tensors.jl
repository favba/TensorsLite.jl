struct AntiSymTen{T, Tyx, Tzx, Tzy} <: AbstractVec{T, 2}
    yx::Tyx
    zx::Tzx
    zy::Tzy

    @inline function AntiSymTen(yx, zx, zy)
        Tyx = typeof(yx)
        Tzx = typeof(zx)
        Tzy = typeof(zy)
        Tf = promote_type_ignoring_Zero(Tyx, Tzx, Tzy)
        yxn = _my_convert(Tf, yx)
        zxn = _my_convert(Tf, zx)
        zyn = _my_convert(Tf, zy)

        Tyxf = typeof(yxn)
        Tzxf = typeof(zxn)
        Tzyf = typeof(zyn)
        Tff = _final_type(Tyxf, Tzxf, Tzyf)
        return new{Tff, Tyxf, Tzxf, Tzyf}(yxn, zxn, zyn)
    end
end

@inline AntiSymTen(; yx = 𝟎, zx = 𝟎, zy = 𝟎) = AntiSymTen(yx, zx, zy)

@inline constructor(::Type{T}) where {T <: AntiSymTen} = AntiSymTen
@inline Base.:+(a::AntiSymTen, b::AntiSymTen) = @inline AntiSymTen(map(+, fields(a), fields(b))...)
@inline Base.:-(a::AntiSymTen, b::AntiSymTen) = @inline AntiSymTen(map(-, fields(a), fields(b))...)
@inline ==(a::AntiSymTen, b::AntiSymTen) = @inline reduce(&, map(==, fields(a), fields(b)))
@inline function _muladd(a::Number, v::AntiSymTen, u::AntiSymTen)
    @inline begin
        at = convert(promote_type(typeof(a), nonzero_eltype(v), nonzero_eltype(u)), a)
        W = AntiSymTen(map(_muladd, ntuple(i -> at, Val(3)), fields(v), fields(u))...)
    end
    return W
end
@inline function _muladd(a::Union{Zero, One}, v::AntiSymTen, u::AntiSymTen)
    @inline begin
        W = AntiSymTen(map(_muladd, ntuple(i -> a, Val(3)), fields(v), fields(u))...)
    end
    return W
end

const AntiSymTen3D{T} = AntiSymTen{T, T, T, T}
const AntiSymTen2Dxy{T} = AntiSymTen{Union{Zero, T}, T, Zero, Zero}
const AntiSymTen2Dxz{T} = AntiSymTen{Union{Zero, T}, Zero, T, Zero}
const AntiSymTen2Dyz{T} = AntiSymTen{Union{Zero, T}, Zero, Zero, T}

const AntiSymTenMaybe2Dxy{T, Tz} = AntiSymTen{Union{T, Tz}, T, Tz, Tz}

Base.IndexStyle(::Type{AntiSymTen}) = IndexCartesian()
@inline function Base.getindex(S::AntiSymTen, i::Integer, j::Integer)
    t = (Int(i), Int(j))
    @boundscheck checkbounds(S, t...)
    t === (1, 1) && return 𝟎
    t === (2, 1) && return S.yx
    t === (1, 2) && return -S.yx
    t === (3, 1) && return S.zx
    t === (1, 3) && return -S.zx
    t === (2, 2) && return 𝟎
    t === (3, 2) && return S.zy
    t === (2, 3) && return -S.zy
    return 𝟎
end

@inline function Base.getproperty(S::AntiSymTen, s::Symbol)
    if s === :x
        xx = 𝟎
        yx = getfield(S, :yx)
        zx = getfield(S, :zx)
        return Vec(xx, yx, zx)
    elseif s === :y
        xy = -getfield(S, :yx)
        yy = 𝟎
        zy = getfield(S, :zy)
        return Vec(xy, yy, zy)
    elseif s === :z
        xz = -getfield(S, :zx)
        yz = -getfield(S, :zy)
        zz = 𝟎
        return Vec(xz, yz, zz)
    elseif s === :xy
        return -getfield(S, :yx)
    elseif s === :xz
        return -getfield(S, :zx)
    elseif s === :yz
        return -getfield(S, :zy)
    elseif s === :xx || s === :yy || s === :zz
        return 𝟎
    else
        return getfield(S, s)
    end
end

@inline transpose(W::AntiSymTen) = -W
@inline adjoint(W::AntiSymTen) = -conj(W)

@inline function Base.convert(::Type{AntiSymTen{T, Tyx, Tzx, Tzy}}, v::AntiSymTen) where {T, Tyx, Tzx, Tzy}
    @inline nfields = map(convert, (Tyx, Tzx, Tzy), fields(v))
    return AntiSymTen(nfields...)
end

@inline function Base.convert(::Type{Vec{T, 2, Vec{_Tx, 1, Txx, Tyx, Tzx}, Vec{_Ty, 1, Txy, Tyy, Tzy}, Vec{_Tz, 1, Txz, Tyz, Tzz}}}, v::AntiSymTen) where {T, _Tx, _Ty, _Tz, Txx, Tyx, Tzx, Txy, Tyy, Tzy, Txz, Tyz, Tzz}
    @inline xx, yx, zx, xy, yy, zy, xz, yz, zz = map(convert, (Txx, Tyx, Tzx, Txy, Tyy, Tzy, Txz, Tyz, Tzz), map(getproperty, (v, v, v, v, v, v, v, v, v), (:xx, :yx, :zx, :xy, :yy, :zy, :xz, :yz, :zz)))
    return Ten(xx, yx, zx, xy, yy, zy, xz, yz, zz)
end

@inline inner(a::AntiSymTen, b::AntiSymTen) = 2 * muladd(a.xy, b.xy, muladd(a.xz, b.xz, a.yz * b.yz))

