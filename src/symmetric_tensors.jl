struct SymTen{T, Txx, Tyx, Tzx, Tyy, Tzy, Tzz} <: AbstractTen{T}
    xx::Txx
    xy::Tyx
    xz::Tzx
    yy::Tyy
    yz::Tzy
    zz::Tzz

    @inline function SymTen(
            xx, xy, xz,
                yy, yz,
                    zz
        )
        Txx = typeof(xx)
        Tyx = typeof(xy)
        Tzx = typeof(xz)
        Tyy = typeof(yy)
        Tzy = typeof(yz)
        Tzz = typeof(zz)
        Tf = promote_type_ignoring_Zero(
            Txx, Tyx, Tzx,
                 Tyy, Tzy,
                      Tzz
        )
        xxn = _my_convert(Tf, xx)
        yxn = _my_convert(Tf, xy)
        zxn = _my_convert(Tf, xz)
        yyn = _my_convert(Tf, yy)
        zyn = _my_convert(Tf, yz)
        zzn = _my_convert(Tf, zz)

        Txxf = typeof(xxn)
        Tyxf = typeof(yxn)
        Tzxf = typeof(zxn)
        Tyyf = typeof(yyn)
        Tzyf = typeof(zyn)
        Tzzf = typeof(zzn)
        Tff = _final_type(Txxf, Tyxf, Tzxf, Tyyf, Tzyf, Tzzf)
        return new{Tff, Txxf, Tyxf, Tzxf, Tyyf, Tzyf, Tzzf}(
            xxn, yxn, zxn,
                 yyn, zyn,
                      zzn
        )
    end
end

@inline SymTen(; xx = 𝟎, xy = 𝟎, xz = 𝟎, yy = 𝟎, yz = 𝟎, zz = 𝟎) = SymTen(xx, xy, xz, yy, yz, zz)

@inline function Base.convert(::Type{SymTen{T, Txx, Tyx, Tzx, Tyy, Tzy, Tzz}}, v::SymTen) where {T, Txx, Tyx, Tzx, Tyy, Tzy, Tzz}
    @inline nfields = map(convert, (Txx, Tyx, Tzx, Tyy, Tzy, Tzz), fields(v))
    return SymTen(nfields...)
end

@inline function Base.convert(::Type{Vec{T, 2, Vec{_Tx, 1, Txx, Tyx, Tzx}, Vec{_Ty, 1, Tyx, Tyy, Tzy}, Vec{_Tz, 1, Tzx, Tzy, Tzz}}}, v::SymTen) where {T, _Tx, _Ty, _Tz, Txx, Tyx, Tzx, Tyy, Tzy, Tzz}
    @inline xx, yx, zx, yy, zy, zz = map(convert, (Txx, Tyx, Tzx, Tyy, Tzy, Tzz), fields(v))

    return Ten(
        xx, yx, zx,
        yx, yy, zy,
        zx, zy, zz
    )
end

@inline constructor(::Type{T}) where {T <: SymTen} = SymTen
@inline Base.:+(a::SymTen, b::SymTen) = @inline SymTen(map(+, fields(a), fields(b))...)
@inline Base.:-(a::SymTen, b::SymTen) = @inline SymTen(map(-, fields(a), fields(b))...)
@inline ==(a::SymTen, b::SymTen) = @inline reduce(&, map(==, fields(a), fields(b)))
@inline function _muladd(a::Number, v::SymTen, u::SymTen)
    @inline begin
        at = convert(promote_type(typeof(a), nonzero_eltype(v), nonzero_eltype(u)), a)
        S = SymTen(map(_muladd, ntuple(i -> at, Val(6)), fields(v), fields(u))...)
    end
    return S
end
@inline function _muladd(a::Union{One, Zero}, v::SymTen, u::SymTen)
    @inline begin
        S = SymTen(map(_muladd, ntuple(i -> a, Val(6)), fields(v), fields(u))...)
    end
    return S
end

const SymTen3D{T} = SymTen{T, T, T, T, T, T, T}
const SymTen2Dxy{T} = SymTen{Union{Zero, T}, T, T, Zero, T, Zero, Zero}
const SymTen2Dxz{T} = SymTen{Union{Zero, T}, T, Zero, T, Zero, Zero, T}
const SymTen2Dyz{T} = SymTen{Union{Zero, T}, Zero, Zero, Zero, T, T, T}
const SymTen2D{T} = Union{SymTen2Dxy{T}, SymTen2Dxz{T}, SymTen2Dyz{T}}
const SymTen1Dx{T} = SymTen{Union{Zero, T}, T, Zero, Zero, Zero, Zero, Zero}
const SymTen1Dy{T} = SymTen{Union{Zero, T}, Zero, Zero, Zero, T, Zero, Zero}
const SymTen1Dz{T} = SymTen{Union{Zero, T}, Zero, Zero, Zero, Zero, Zero, T}
const SymTen1D{T} = Union{SymTen1Dx{T}, SymTen1Dy{T}, SymTen1Dz{T}}

const SymTenMaybe2Dxy{T, Tz} = SymTen{Union{T, Tz}, T, T, Tz, T, Tz, Tz}

Base.IndexStyle(::Type{SymTen}) = IndexCartesian()
@inline function Base.getindex(S::SymTen, i::Integer, j::Integer)
    t = (Int(i), Int(j))
    @boundscheck checkbounds(S, t...)
    t === (1, 1) && return S.xx
    (t === (2, 1) || t === (1, 2)) && return S.xy
    (t === (3, 1) || t === (1, 3)) && return S.xz
    t === (2, 2) && return S.yy
    (t === (3, 2) || t === (2, 3))  && return S.yz
    return S.zz
end

@inline function Base.getproperty(S::SymTen, s::Symbol)
    if s === :x
        xx = getfield(S, :xx)
        yx = getfield(S, :xy)
        zx = getfield(S, :xz)
        return Vec(xx, yx, zx)
    elseif s === :y
        xy = getfield(S, :xy)
        yy = getfield(S, :yy)
        zy = getfield(S, :yz)
        return Vec(xy, yy, zy)
    elseif s === :z
        xz = getfield(S, :xz)
        yz = getfield(S, :yz)
        zz = getfield(S, :zz)
        return Vec(xz, yz, zz)
    elseif s === :yx
        return getfield(S, :xy)
    elseif s === :zx
        return getfield(S, :xz)
    elseif s === :zy
        return getfield(S, :yz)
    else
        return getfield(S, s)
    end
end

@inline transpose(S::SymTen) = S
@inline adjoint(S::SymTen) = conj(transpose(S))

@inline inner(a::SymTen, b::SymTen) =  muladd(2, muladd(a.xy, b.xy, muladd(a.xz, b.xz, a.yz*b.yz)), muladd(a.xx, b.xx, muladd(a.yy, b.yy, a.zz*b.zz)))

