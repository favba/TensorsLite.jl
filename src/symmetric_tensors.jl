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

@inline function Base.convert(::Type{Tensor{T, 2, Tensor{_Tx, 1, Txx, Tyx, Tzx}, Tensor{_Ty, 1, Tyx, Tyy, Tzy}, Tensor{_Tz, 1, Tzx, Tzy, Tzz}}}, v::SymTen) where {T, _Tx, _Ty, _Tz, Txx, Tyx, Tzx, Tyy, Tzy, Tzz}
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

SymTen3D{T}(xx, xy, xz, yy, yz, zz) where {T} = SymTen(convert(T, xx), convert(T, xy), convert(T, xz),
                                                                       convert(T, yy), convert(T, yz),
                                                                                       convert(T, zz))

SymTen3D(xx, xy, xz, yy, yz, zz) = SymTen3D{promote_type(typeof(xx), typeof(xy), typeof(xz),
                                                                     typeof(yy), typeof(yz),
                                                                                 typeof(zz))}(xx, xy, xz,
                                                                                                  yy, yz,
                                                                                                      zz)


const SymTen2Dxy{T} = SymTen{Union{Zero, T}, T, T, Zero, T, Zero, Zero}

SymTen2Dxy{T}(xx, xy, yy) where {T} = SymTen(convert(T, xx), convert(T, xy), Zero(),
                                                             convert(T, yy), Zero(),
                                                                             Zero())

SymTen2Dxy(xx, xy, yy) = SymTen2Dxy{promote_type(typeof(xx), typeof(xy), typeof(yy))}(xx, xy, yy)


const SymTen2Dxz{T} = SymTen{Union{Zero, T}, T, Zero, T, Zero, Zero, T}

SymTen2Dxz{T}(xx, xz, zz) where {T} = SymTen(convert(T, xx), Zero(), convert(T, xz),
                                                             Zero(), Zero(),
                                                                     convert(T, zz))

SymTen2Dxz(xx, xz, zz) = SymTen2Dxz{promote_type(typeof(xx), typeof(xz), typeof(zz))}(xx, xz, zz)


const SymTen2Dyz{T} = SymTen{Union{Zero, T}, Zero, Zero, Zero, T, T, T}

SymTen2Dyz{T}(yy, yz, zz) where {T} = SymTen(Zero(), Zero(),         Zero() ,
                                                     convert(T, yy), convert(T, yz),
                                                                     convert(T, zz))

SymTen2Dyz(yy, yz, zz) = SymTen2Dyz{promote_type(typeof(yy), typeof(yz), typeof(zz))}(yy, yz, zz)


const SymTen2D{T} = Union{SymTen2Dxy{T}, SymTen2Dxz{T}, SymTen2Dyz{T}}


const SymTen1Dx{T} = SymTen{Union{Zero, T}, T, Zero, Zero, Zero, Zero, Zero}

SymTen1Dx{T}(xx) where {T} = SymTen(convert(T, xx), Zero(), Zero(),
                                                    Zero(), Zero(),
                                                            Zero())

SymTen1Dx(xx) = SymTen1Dx{typeof(xx)}(xx)


const SymTen1Dy{T} = SymTen{Union{Zero, T}, Zero, Zero, Zero, T, Zero, Zero}

SymTen1Dy{T}(yy) where {T} = SymTen(Zero(), Zero(),         Zero(),
                                            convert(T, yy), Zero(),
                                                            Zero())

SymTen1Dy(yy) = SymTen1Dy{typeof(yy)}(yy)


const SymTen1Dz{T} = SymTen{Union{Zero, T}, Zero, Zero, Zero, Zero, Zero, T}
SymTen1Dz{T}(zz) where {T} = SymTen(Zero(), Zero(), Zero(),
                                            Zero(), Zero(),
                                                    convert(T, zz))

SymTen1Dz(zz) = SymTen1Dz{typeof(zz)}(zz)


const SymTen1D{T} = Union{SymTen1Dx{T}, SymTen1Dy{T}, SymTen1Dz{T}}

const SymTenMaybe2Dxy{T, Tz} = SymTen{Union{T, Tz}, T, T, Tz, T, Tz, Tz}

Base.IndexStyle(::Type{SymTen}) = IndexCartesian()
Base.@constprop :aggressive function Base.getindex(S::SymTen, i::Integer, j::Integer)
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
        return Tensor(xx, yx, zx)
    elseif s === :y
        xy = getfield(S, :xy)
        yy = getfield(S, :yy)
        zy = getfield(S, :yz)
        return Tensor(xy, yy, zy)
    elseif s === :z
        xz = getfield(S, :xz)
        yz = getfield(S, :yz)
        zz = getfield(S, :zz)
        return Tensor(xz, yz, zz)
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

@inline inner(a::SymTen{T1}, b::SymTen{T2}) where {T1<:Real, T2<:Real} =  muladd(2, muladd(a.xy, b.xy, muladd(a.xz, b.xz, a.yz*b.yz)), muladd(a.xx, b.xx, muladd(a.yy, b.yy, a.zz*b.zz)))

Base.rand(::Type{SymTen{T,Txx,Txy,Txz,Tyy,Tyz,Tzz}}) where {T,Txx,Txy,Txz,Tyy,Tyz,Tzz} = SymTen(rand(Txx), rand(Txy), rand(Txz), rand(Tyy), rand(Tyz), rand(Tzz))
