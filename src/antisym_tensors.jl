
abstract type AbstractAntiSymmetricTensor{N,T} <: AbstractTensor{N, T} end

const AntiSymTen{T} = AbstractAntiSymmetricTensor{2,T}

struct AntiSymmetricTensor{N, T, Tyx, Tzx, Tzy} <: AbstractAntiSymmetricTensor{N, T}
    xy::Tyx
    xz::Tzx
    yz::Tzy

    @inline function AntiSymmetricTensor(xy, xz, yz)

        # AbstractTensor are only valid as input when they are all of the same order
        # Same order Tensors as input are dealt with in the next method definition
        xy isa AbstractTensor && throw(DimensionMismatch())
        xz isa AbstractTensor && throw(DimensionMismatch())
        yz isa AbstractTensor && throw(DimensionMismatch())

        Tyx = typeof(xy)
        Tzx = typeof(xz)
        Tzy = typeof(yz)
        Tf = promote_type_ignoring_Zero_and_One(Tyx, Tzx, Tzy)
        yxn = _my_convert(Tf, xy)
        zxn = _my_convert(Tf, xz)
        zyn = _my_convert(Tf, yz)

        Tyxf = typeof(yxn)
        Tzxf = typeof(zxn)
        Tzyf = typeof(zyn)
        Tff = Union{Tyxf, Tzxf, Tzyf}
        return new{2, Union{Tff, Zero}, Tyxf, Tzxf, Tzyf}(yxn, zxn, zyn)
    end

    @inline function AntiSymmetricTensor(xy::AbstractTensor{N,Txy}, xz::AbstractTensor{N,Txz}, yz::AbstractTensor{N,Tyz}) where {N, Txy, Txz, Tyz}
        Tf = promote_type_ignoring_Zero_and_One(_non_StaticBool_type(Txy), _non_StaticBool_type(Txz), _non_StaticBool_type(Tyz))
        xyf = _eltype_convert(Tf, xy)
        xzf = _eltype_convert(Tf, xz)
        yzf = _eltype_convert(Tf, yz)
        return new{N+2, Union{eltype(xyf), eltype(xzf), eltype(yzf), Zero}, typeof(xyf), typeof(xzf), typeof(yzf)}(xyf, xzf, yzf)
    end

end

@inline AntiSymTen(xy, xz, yz) = AntiSymmetricTensor(xy, xz, yz)
@inline AntiSymTen(; xy = 𝟎, xz = 𝟎, yz = 𝟎) = AntiSymmetricTensor(xy, xz, yz)

@inline constructor(::Type{T}) where {T <: AntiSymmetricTensor} = AntiSymmetricTensor
@inline Base.:+(a::AntiSymmetricTensor, b::AntiSymmetricTensor) = @inline AntiSymmetricTensor(map(+, fields(a), fields(b))...)
@inline Base.:-(a::AntiSymmetricTensor, b::AntiSymmetricTensor) = @inline AntiSymmetricTensor(map(-, fields(a), fields(b))...)
@inline ==(a::AntiSymmetricTensor, b::AntiSymmetricTensor) = @inline reduce(&, map(==, fields(a), fields(b)))
@inline function _muladd(a::Number, v::AntiSymmetricTensor, u::AntiSymmetricTensor)
    @inline begin
        at = convert(promote_type(typeof(a), nonzero_eltype(v), nonzero_eltype(u)), a)
        W = AntiSymmetricTensor(map(_muladd, ntuple(i -> at, Val(3)), fields(v), fields(u))...)
    end
    return W
end

const AntiSymTen3D{T} = AntiSymmetricTensor{2, Union{Zero,T}, T, T, T}

AntiSymTen3D{T}(xy, xz, yz) where {T} = AntiSymmetricTensor(convert(T, xy), convert(T, xz), convert(T, yz))

AntiSymTen3D(xy, xz, yz) = AntiSymTen3D{promote_type(typeof(xy), typeof(xz), typeof(yz))}(xy, xz, yz)


const AntiSymTen2Dxy{T} = AntiSymmetricTensor{2, Union{Zero, T}, T, Zero, Zero}

AntiSymTen2Dxy{T}(xy) where {T} = AntiSymmetricTensor(convert(T, xy), Zero(), Zero())

AntiSymTen2Dxy(xy) = AntiSymTen2Dxy{typeof(xy)}(xy)


const AntiSymTen2Dxz{T} = AntiSymmetricTensor{2, Union{Zero, T}, Zero, T, Zero}

AntiSymTen2Dxz{T}(xz) where {T} = AntiSymmetricTensor(Zero(), convert(T, xz), Zero())

AntiSymTen2Dxz(xz) = AntiSymTen2Dxz{typeof(xz)}(xz)


const AntiSymTen2Dyz{T} = AntiSymmetricTensor{2, Union{Zero, T}, Zero, Zero, T}

AntiSymTen2Dyz{T}(yz) where {T} = AntiSymmetricTensor(Zero(), Zero(), convert(T, yz))

AntiSymTen2Dyz(yz) = AntiSymTen2Dyz{typeof(yz)}(yz)


const AntiSymTenMaybe2Dxy{T, Tz} = AntiSymmetricTensor{2, Union{T, Zero}, T, Tz, Tz}

Base.IndexStyle(::Type{AntiSymmetricTensor}) = IndexCartesian()

Base.@constprop :aggressive function Base.getindex(S::AntiSymmetricTensor{N}, I::Vararg{Integer,N}) where {N}

    tI = map(Int, I)
    @boundscheck checkbounds(S, tI...)
    t = (tI[N-1], tI[N])
    mtI = tI[Base.OneTo(N-2)]

    zt = 𝟎

    t === (1, 1) && return zt
    t === (2, 1) && return @inbounds(S.yx[mtI...])
    t === (1, 2) && return -@inbounds(S.yx[mtI...])
    t === (3, 1) && return @inbounds(S.zx[mtI...])
    t === (1, 3) && return -@inbounds(S.zx[mtI...])
    t === (2, 2) && return zt
    t === (3, 2) && return @inbounds(S.zy[mtI...])
    t === (2, 3) && return -@inbounds(S.zy[mtI...])
    return zt
end

@inline function Base.getproperty(S::AntiSymmetricTensor{N}, s::Symbol) where {N}
    zt = Tensor{N-2}()
    if s === :x
        xx = zt
        yx = -getfield(S, :xy)
        zx = -getfield(S, :xz)
        return Tensor(xx, yx, zx)
    elseif s === :y
        xy = getfield(S, :xy)
        yy = zt
        zy = -getfield(S, :yz)
        return Tensor(xy, yy, zy)
    elseif s === :z
        xz = getfield(S, :xz)
        yz = getfield(S, :yz)
        zz = zt
        return Tensor(xz, yz, zz)
    elseif s === :yx
        return -getfield(S, :xy)
    elseif s === :zx
        return -getfield(S, :xz)
    elseif s === :zy
        return -getfield(S, :yz)
    elseif s === :xx || s === :yy || s === :zz
        return zt
    else
        return getfield(S, s)
    end
end

@inline function Base.convert(::Type{AntiSymmetricTensor{N, T, Tyx, Tzx, Tzy}}, v::AntiSymmetricTensor{N}) where {N, T, Tyx, Tzx, Tzy}
    @inline nfields = map(convert, (Tyx, Tzx, Tzy), fields(v))
    return AntiSymmetricTensor(nfields...)
end

@inline function Base.convert(::Type{Tensor{N, T, Tensor{<:Any, _Tx, Txx, Tyx, Tzx}, Tensor{<:Any, _Ty, Txy, Tyy, Tzy}, Tensor{<:Any, _Tz, Txz, Tyz, Tzz}}}, v::AntiSymmetricTensor{N}) where {N,T, _Tx, _Ty, _Tz, Txx, Tyx, Tzx, Txy, Tyy, Tzy, Txz, Tyz, Tzz}
    @inline xx, yx, zx, xy, yy, zy, xz, yz, zz = map(convert, (Txx, Tyx, Tzx, Txy, Tyy, Tzy, Txz, Tyz, Tzz), map(getproperty, (v, v, v, v, v, v, v, v, v), (:xx, :yx, :zx, :xy, :yy, :zy, :xz, :yz, :zz)))
    return Ten(xx, xy, xz,
               yx, yy, yz,
               zx, zy, zz)
end

@inline inner(a::AntiSymmetricTensor{2, <:Real}, b::AntiSymmetricTensor{2, <:Real}) = 2 * _muladd(a.xy, b.xy, _muladd(a.xz, b.xz, a.yz * b.yz))

@inline inneradd(a::AntiSymmetricTensor{2, <:Real}, b::AntiSymmetricTensor{2, <:Real}, c::Real) = _muladd(2, _muladd(a.xy, b.xy, _muladd(a.xz, b.xz, a.yz * b.yz)), c)

@inline inner(::AntiSymmetricTensor{2}, ::SymmetricTensor{2}) = 𝟎
@inline inner(::SymmetricTensor{2}, ::AntiSymmetricTensor{2}) = 𝟎

@inline inneradd(::AntiSymmetricTensor{2}, ::SymmetricTensor{2}, c) = c
@inline inneradd(::SymmetricTensor{2}, ::AntiSymmetricTensor{2}, c) = c

Base.rand(::Type{AntiSymmetricTensor{N, T,Txy,Txz,Tyz}}) where {N, T,Txy,Txz,Tyz} = AntiSymmetricTensor(rand(Txy), rand(Txz), rand(Tyz))

