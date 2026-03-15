abstract type AbstractSymmetricTensor{N, T} <: AbstractTensor{N, T} end

const SymTen{T} = AbstractSymmetricTensor{2,T}

struct SymmetricTensor{N, T, Txx, Tyx, Tzx, Tyy, Tzy, Tzz} <: AbstractSymmetricTensor{N, T}
    xx::Txx
    xy::Tyx
    xz::Tzx
    yy::Tyy
    yz::Tzy
    zz::Tzz

    @inline function SymmetricTensor(
            xx, xy, xz,
                yy, yz,
                    zz
        )

        mapreduce(x->isa(x,AbstractTensor), |, (xx,xy,xz,yy,yz,zz)) && throw(DimensionMismatch())

        Txx = typeof(xx)
        Tyx = typeof(xy)
        Tzx = typeof(xz)
        Tyy = typeof(yy)
        Tzy = typeof(yz)
        Tzz = typeof(zz)
        Tf = promote_type_ignoring_Zero_and_One(
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
        Tff = Union{Txxf, Tyxf, Tzxf, Tyyf, Tzyf, Tzzf}
        return new{2, Tff, Txxf, Tyxf, Tzxf, Tyyf, Tzyf, Tzzf}(
            xxn, yxn, zxn,
                 yyn, zyn,
                      zzn
        )
    end

    @inline function SymmetricTensor(
            xx::AbstractTensor{N,Txx}, xy::AbstractTensor{N,Txy}, xz::AbstractTensor{N,Txz},
                yy::AbstractTensor{N,Tyy}, yz::AbstractTensor{N,Tyz},
                    zz::AbstractTensor{N,Tzz}
        ) where {N, Txx, Txy, Txz, Tyy, Tyz, Tzz}

        Tf = promote_type_ignoring_Zero_and_One(map(_non_StaticBool_type,(Txx,Txy,Txz,Tyy,Tyz,Tzz))...)

        xxf = _eltype_convert(Tf,xx)
        xyf = _eltype_convert(Tf,xy)
        xzf = _eltype_convert(Tf,xz)
        yyf = _eltype_convert(Tf,yy)
        yzf = _eltype_convert(Tf,yz)
        zzf = _eltype_convert(Tf,zz)

        return new{
                   N+2,
                   Union{map(eltype,(xxf,xyf,xzf,yyf,yzf,zzf))...},
                   map(typeof,(xxf,xyf,xzf,yyf,yzf,zzf))...
               }(xxf,xyf,xzf,yyf,yzf,zzf)
    end

end

@inline constructor(::Type{T}) where {T <: SymmetricTensor} = SymmetricTensor

@inline SymmetricTensor{2}() = SymmetricTensor(Zero(),Zero(),Zero(),Zero(),Zero(),Zero())

@inline function SymmetricTensor(; xx = 𝟎, xy = 𝟎, xz = 𝟎, yy = 𝟎, yz = 𝟎, zz = 𝟎)
    if (xx === 𝟎) && (xy == 𝟎) && (xz == 𝟎) && (yy === 𝟎) && (yz == 𝟎) && (zz == 𝟎)
        return SymmetricTensor{2}()
    else
        check_args_ignoring_zeros(xx,xy,xz,yy,yz,zz)
        NV = _get_ndims(xx,xy,xz,yy,yz,zz)
        xxf, xyf, xzf, yyf, yzf, zzf = if_zero_to_tensor(NV,xx,xy,xz,yy,yz,zz)
        return SymmetricTensor(xxf, xyf, xzf, yyf, yzf, zzf)
    end
end

@inline function Base.convert(::Type{SymmetricTensor{N, T, Txx, Tyx, Tzx, Tyy, Tzy, Tzz}}, v::SymmetricTensor{N}) where {N, T, Txx, Tyx, Tzx, Tyy, Tzy, Tzz}
    @inline nfields = map(convert, (Txx, Tyx, Tzx, Tyy, Tzy, Tzz), fields(v))
    return SymmetricTensor(nfields...)
end

@inline Base.:+(a::SymmetricTensor{N}, b::SymmetricTensor{N}) where {N} = @inline SymmetricTensor(map(+, fields(a), fields(b))...)
@inline Base.:-(a::SymmetricTensor{N}, b::SymmetricTensor{N}) where {N} = @inline SymmetricTensor(map(-, fields(a), fields(b))...)
@inline ==(a::SymmetricTensor{N}, b::SymmetricTensor{N}) where {N} = @inline reduce(&, map(==, fields(a), fields(b)))
@inline function _muladd(a::Number, v::SymmetricTensor{N}, u::SymmetricTensor{N}) where {N}
    @inline begin
        at = convert(promote_type(typeof(a), nonzero_eltype(v), nonzero_eltype(u)), a)
        S = SymmetricTensor(map(_muladd, ntuple(i -> at, Val(6)), fields(v), fields(u))...)
    end
    return S
end

@inline _muladd(::Zero, ::SymmetricTensor{N}, S::SymmetricTensor{N}) where {N} = S

@inline function SymTen(a, b, c, d, e, f)
    if (a isa AbstractTensor || b isa AbstractTensor || c isa AbstractTensor ||
        d isa AbstractTensor || e isa AbstractTensor || f isa AbstractTensor)
        throw(ArgumentError("Tensors are not valid input to the `SymTen` function"))
    end
    return SymmetricTensor(xx=a, xy=b, xz=c, yy=d, yz=e, zz=f)
end

@inline SymTen(; xx = 𝟎, xy = 𝟎, xz = 𝟎, yy = 𝟎, yz = 𝟎, zz = 𝟎) = SymTen(xx,xy,xz,yy,yz,zz)

const SymTen3D{T} = SymmetricTensor{2, T, T, T, T, T, T, T}

SymTen3D{T}(xx, xy, xz, yy, yz, zz) where {T} = SymTen(convert(T, xx), convert(T, xy), convert(T, xz),
                                                                       convert(T, yy), convert(T, yz),
                                                                                       convert(T, zz))

SymTen3D(xx, xy, xz, yy, yz, zz) = SymTen3D{promote_type(typeof(xx), typeof(xy), typeof(xz),
                                                                     typeof(yy), typeof(yz),
                                                                                 typeof(zz))}(xx, xy, xz,
                                                                                                  yy, yz,
                                                                                                      zz)


const SymTen2Dxy{T} = SymmetricTensor{2, Union{Zero, T}, T, T, Zero, T, Zero, Zero}

SymTen2Dxy{T}(xx, xy, yy) where {T} = SymTen(convert(T, xx), convert(T, xy), Zero(),
                                                             convert(T, yy), Zero(),
                                                                             Zero())

SymTen2Dxy(xx, xy, yy) = SymTen2Dxy{promote_type(typeof(xx), typeof(xy), typeof(yy))}(xx, xy, yy)


const SymTen2Dxz{T} = SymmetricTensor{2, Union{Zero, T}, T, Zero, T, Zero, Zero, T}

SymTen2Dxz{T}(xx, xz, zz) where {T} = SymTen(convert(T, xx), Zero(), convert(T, xz),
                                                             Zero(), Zero(),
                                                                     convert(T, zz))

SymTen2Dxz(xx, xz, zz) = SymTen2Dxz{promote_type(typeof(xx), typeof(xz), typeof(zz))}(xx, xz, zz)


const SymTen2Dyz{T} = SymmetricTensor{2, Union{Zero, T}, Zero, Zero, Zero, T, T, T}

SymTen2Dyz{T}(yy, yz, zz) where {T} = SymTen(Zero(), Zero(),         Zero() ,
                                                     convert(T, yy), convert(T, yz),
                                                                     convert(T, zz))

SymTen2Dyz(yy, yz, zz) = SymTen2Dyz{promote_type(typeof(yy), typeof(yz), typeof(zz))}(yy, yz, zz)


const SymTen2D{T} = Union{SymTen2Dxy{T}, SymTen2Dxz{T}, SymTen2Dyz{T}}


const SymTen1Dx{T} = SymmetricTensor{2, Union{Zero, T}, T, Zero, Zero, Zero, Zero, Zero}

SymTen1Dx{T}(xx) where {T} = SymTen(convert(T, xx), Zero(), Zero(),
                                                    Zero(), Zero(),
                                                            Zero())

SymTen1Dx(xx) = SymTen1Dx{typeof(xx)}(xx)


const SymTen1Dy{T} = SymmetricTensor{2, Union{Zero, T}, Zero, Zero, Zero, T, Zero, Zero}

SymTen1Dy{T}(yy) where {T} = SymTen(Zero(), Zero(),         Zero(),
                                            convert(T, yy), Zero(),
                                                            Zero())

SymTen1Dy(yy) = SymTen1Dy{typeof(yy)}(yy)


const SymTen1Dz{T} = SymmetricTensor{2, Union{Zero, T}, Zero, Zero, Zero, Zero, Zero, T}

SymTen1Dz{T}(zz) where {T} = SymTen(Zero(), Zero(), Zero(),
                                            Zero(), Zero(),
                                                    convert(T, zz))

SymTen1Dz(zz) = SymTen1Dz{typeof(zz)}(zz)


const SymTen1D{T} = Union{SymTen1Dx{T}, SymTen1Dy{T}, SymTen1Dz{T}}

const SymTenMaybe2Dxy{T, Tz} = SymmetricTensor{2, Union{T, Tz}, T, T, Tz, T, Tz, Tz}

Base.IndexStyle(::Type{SymmetricTensor}) = IndexCartesian()

Base.@constprop :aggressive function Base.getindex(S::SymmetricTensor{N}, I::Vararg{Integer,N}) where {N}
    tI = map(Int, I)
    @boundscheck _checkbounds(S, tI...)
    t = (tI[N-1], tI[N])
    mtI = tI[Base.OneTo(N-2)]
    t === (1, 1) && return @inbounds(S.xx[mtI...])
    (t === (2, 1) || t === (1, 2)) && return @inbounds(S.xy[mtI...])
    (t === (3, 1) || t === (1, 3)) && return @inbounds(S.xz[mtI...])
    t === (2, 2) && return @inbounds(S.yy[mtI...])
    (t === (3, 2) || t === (2, 3))  && return @inbounds(S.yz[mtI...])
    return @inbounds(S.zz[mtI...])
end

@inline function Base.getproperty(S::SymmetricTensor, s::Symbol)
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

@inline inner(a::SymmetricTensor{2,<:Real}, b::SymmetricTensor{2,<:Real}) =  _muladd(2, _muladd(a.xy, b.xy, _muladd(a.xz, b.xz, a.yz*b.yz)), _muladd(a.xx, b.xx, _muladd(a.yy, b.yy, a.zz*b.zz)))

@inline inneradd(a::SymmetricTensor{2,<:Real}, b::SymmetricTensor{2,<:Real}, c::Real) =  _muladd(2, _muladd(a.xy, b.xy, _muladd(a.xz, b.xz, a.yz*b.yz)), _muladd(a.xx, b.xx, _muladd(a.yy, b.yy, _muladd(a.zz,b.zz, c))))

Base.rand(::Type{SymmetricTensor{N,T,Txx,Txy,Txz,Tyy,Tyz,Tzz}}) where {N,T,Txx,Txy,Txz,Tyy,Tyz,Tzz} = SymmetricTensor(rand(Txx), rand(Txy), rand(Txz), rand(Tyy), rand(Tyz), rand(Tzz))
