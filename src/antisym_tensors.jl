struct AntiSymTen{T, Tyx, Tzx, Tzy} <: AbstractVec{T, 2}
    xy::Tyx
    xz::Tzx
    yz::Tzy

    @inline function AntiSymTen(xy, xz, yz)
        Tyx = typeof(xy)
        Tzx = typeof(xz)
        Tzy = typeof(yz)
        Tf = promote_type_ignoring_Zero(Tyx, Tzx, Tzy)
        yxn = _my_convert(Tf, xy)
        zxn = _my_convert(Tf, xz)
        zyn = _my_convert(Tf, yz)

        Tyxf = typeof(yxn)
        Tzxf = typeof(zxn)
        Tzyf = typeof(zyn)
        Tff = _final_type(Tyxf, Tzxf, Tzyf)
        return new{Tff, Tyxf, Tzxf, Tzyf}(yxn, zxn, zyn)
    end
end

@inline AntiSymTen(; xy = 𝟎, xz = 𝟎, yz = 𝟎) = AntiSymTen(xy, xz, yz)

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

AntiSymTen3D{T}(xy, xz, yz) where {T} = AntiSymTen(convert(T, xy), convert(T, xz), convert(T, yz))

AntiSymTen3D(xy, xz, yz) = AntiSymTen3D{promote_type(typeof(xy), typeof(xz), typeof(yz))}(xy, xz, yz)


const AntiSymTen2Dxy{T} = AntiSymTen{Union{Zero, T}, T, Zero, Zero}

AntiSymTen2Dxy{T}(xy) where {T} = AntiSymTen(convert(T, xy), Zero(), Zero())

AntiSymTen2Dxy(xy) = AntiSymTen2Dxy{typeof(xy)}(xy)


const AntiSymTen2Dxz{T} = AntiSymTen{Union{Zero, T}, Zero, T, Zero}

AntiSymTen2Dxz{T}(xz) where {T} = AntiSymTen(Zero(), convert(T, xz), Zero())

AntiSymTen2Dxz(xz) = AntiSymTen2Dxz{typeof(xz)}(xz)


const AntiSymTen2Dyz{T} = AntiSymTen{Union{Zero, T}, Zero, Zero, T}

AntiSymTen2Dyz{T}(yz) where {T} = AntiSymTen(Zero(), Zero(), convert(T, yz))

AntiSymTen2Dyz(yz) = AntiSymTen2Dyz{typeof(yz)}(yz)


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
        yx = -getfield(S, :xy)
        zx = -getfield(S, :xz)
        return Vec(xx, yx, zx)
    elseif s === :y
        xy = getfield(S, :xy)
        yy = 𝟎
        zy = -getfield(S, :yz)
        return Vec(xy, yy, zy)
    elseif s === :z
        xz = getfield(S, :xz)
        yz = getfield(S, :yz)
        zz = 𝟎
        return Vec(xz, yz, zz)
    elseif s === :yx
        return -getfield(S, :xy)
    elseif s === :zx
        return -getfield(S, :xz)
    elseif s === :zy
        return -getfield(S, :yz)
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
    return Ten(xx, xy, xz,
               yx, yy, yz,
               zx, zy, zz)
end

@inline inner(a::AntiSymTen{T1}, b::AntiSymTen{T2}) where {T1<:Real, T2<:Real} = 2 * muladd(a.xy, b.xy, muladd(a.xz, b.xz, a.yz * b.yz))

@inline inner(::AntiSymTen, ::SymTen) = 𝟎
@inline inner(::SymTen, ::AntiSymTen) = 𝟎

Base.rand(::Type{AntiSymTen{T,Txy,Txz,Tyz}}) where {T,Txy,Txz,Tyz} = AntiSymTen(rand(Txy), rand(Txz), rand(Tyz))

