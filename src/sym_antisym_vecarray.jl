struct SymTenArray{T, N, Txx, Tyx, Tzx, Tyy, Tzy, Tzz} <: AbstractArray{T, N}
    xx::Txx
    xy::Tyx
    xz::Tzx
    yy::Tyy
    yz::Tzy
    zz::Tzz

    SymTenArray{T}(I::Vararg{Int, N}) where {T, N} = new{SymTen{T, T, T, T, T, T, T}, N, Array{T, N}, Array{T, N}, Array{T, N}, Array{T, N}, Array{T, N}, Array{T, N}}(
        Array{T}(undef, I...), Array{T}(undef, I...), Array{T}(undef, I...),
        Array{T}(undef, I...), Array{T}(undef, I...), Array{T}(undef, I...)
    )

    function SymTenArray(
            xx::AbstractArray, yx::AbstractArray, zx::AbstractArray,
                               yy::AbstractArray, zy::AbstractArray,
                                                  zz::AbstractArray
        )

        s = size(xx)
        N = ndims(xx)
        size(yx) === s || throw(DimensionMismatch("Input Arrays must have the same size"))
        size(zx) === s || throw(DimensionMismatch("Input Arrays must have the same size"))
        size(yy) === s || throw(DimensionMismatch("Input Arrays must have the same size"))
        size(zy) === s || throw(DimensionMismatch("Input Arrays must have the same size"))
        size(zz) === s || throw(DimensionMismatch("Input Arrays must have the same size"))


        Txx = eltype(xx)
        Tyx = eltype(yx)
        Tzx = eltype(zx)
        Tyy = eltype(yy)
        Tzy = eltype(zy)
        Tzz = eltype(zz)
        Tf = promote_type_ignoring_Zero(
            Txx, Tyx, Tzx,
                 Tyy, Tzy,
                      Tzz
        )


        Txxf = _my_promote_type(Tf, Txx)
        Tyxf = _my_promote_type(Tf, Tyx)
        Tzxf = _my_promote_type(Tf, Tzx)
        Tyyf = _my_promote_type(Tf, Tyy)
        Tzyf = _my_promote_type(Tf, Tzy)
        Tzzf = _my_promote_type(Tf, Tzz)
        Tff = _final_type(Txxf, Tyxf, Tzxf, Tyyf, Tzyf, Tzzf)

        return new{
            SymTen{Tff, Txxf, Tyxf, Tzxf, Tyyf, Tzyf, Tzzf}, N,
            typeof(xx), typeof(yx), typeof(zx),
                        typeof(yy), typeof(zy),
                                    typeof(zz)
        }(
            xx, yx, zx,
                yy, zy,
                    zz
        )
    end

end

function SymTenArray(
        xx, yx, zx,
            yy, zy,
                zz
    )

    vals = (
        xx, yx, zx,
            yy, zy,
                zz,
    )

    vals === (
        𝟎, 𝟎, 𝟎,
           𝟎, 𝟎,
              𝟎,
    ) && throw(DomainError(vals, "At least one entry must be a valid Array"))

    non_zero_vals = _filter_zeros(vals...)
    s = size(non_zero_vals[1])

    all(x -> (size(x) === s), non_zero_vals) || throw(DimensionMismatch("Input Arrays must have the same size"))

    sizes = (s, s, s, s, s, s)
    final_vals = map(_if_zero_to_Array, sizes, vals)

    return SymTenArray(final_vals...)
end

SymTenArray(;
    xx::Union{Zero, <:AbstractArray} = 𝟎, xy::Union{Zero, <:AbstractArray} = 𝟎, xz::Union{Zero, <:AbstractArray} = 𝟎,
                                          yy::Union{Zero, <:AbstractArray} = 𝟎, yz::Union{Zero, <:AbstractArray} = 𝟎,
                                                                                zz::Union{Zero, <:AbstractArray} = 𝟎
) = SymTenArray(
    xx, xy, xz,
        yy, yz,
            zz
)

const SymTen3DArray{T, N} = SymTenArray{
    SymTen3D{T}, N, Array{T, N}, Array{T, N}, Array{T, N},
                                 Array{T, N}, Array{T, N},
                                              Array{T, N}
}


SymTen3DArray(a::AbstractArray{T,N}, b::AbstractArray{T,N}, c::AbstractArray{T,N},
                                     d::AbstractArray{T,N}, e::AbstractArray{T,N},
                                                            f::AbstractArray{T,N}) where {T,N} =
    SymTenArray(a, b, c,
                   d, e,
                      f)


const SymTen2DxyArray{T, N} = SymTenArray{
    SymTen2Dxy{T}, N, Array{T, N}, Array{T, N}, Array{Zero, N},
                                   Array{T, N}, Array{Zero, N},
                                                Array{Zero, N}
}

SymTen2DxyArray(a::AbstractArray{T,N}, b::AbstractArray{T,N},
                d::AbstractArray{T,N}) where {T,N} =
    SymTenArray(a, b, ZeroArray(size(a)),
                   d, ZeroArray(size(a)),
                      ZeroArray(size(a)))


const SymTen2DxzArray{T, N} = SymTenArray{
    SymTen2Dxz{T}, N, Array{T, N}, Array{Zero, N}, Array{T, N},
                                   Array{Zero, N}, Array{Zero, N},
                                                   Array{T, N}
}

SymTen2DxzArray(a::AbstractArray{T,N}, c::AbstractArray{T,N},
                                       f::AbstractArray{T,N}) where {T,N} =
    SymTenArray(a, ZeroArray(size(a)), c,
                   ZeroArray(size(a)), ZeroArray(size(a)),
                                       f)


const SymTen2DyzArray{T, N} = SymTenArray{
    SymTen2Dyz{T}, N, Array{Zero, N}, Array{Zero, N}, Array{Zero, N},
                                      Array{T, N},    Array{T, N},
                                                      Array{T, N}
}

SymTen2DyzArray(d::AbstractArray{T,N}, e::AbstractArray{T,N},
                                       f::AbstractArray{T,N}) where {T,N} =
    SymTenArray(ZeroArray(size(d)), ZeroArray(size(d)), ZeroArray(size(d)),
                d,                  e,
                                                        f)


const SymTen1DxArray{T, N} = SymTenArray{
    SymTen1Dx{T}, N, Array{T, N}, Array{Zero, N}, Array{Zero, N},
                                  Array{Zero, N}, Array{Zero, N},
                                                  Array{Zero, N}
}

SymTen1DxArray(a::AbstractArray{T,N}) where {T,N} =
    SymTenArray(a, ZeroArray(size(a)), ZeroArray(size(a)),
                   ZeroArray(size(a)), ZeroArray(size(a)),
                                       ZeroArray(size(a)))


const SymTen1DyArray{T, N} = SymTenArray{
    SymTen1Dy{T}, N, Array{Zero, N}, Array{Zero, N}, Array{Zero, N},
                                     Array{T, N},    Array{Zero, N},
                                                     Array{Zero, N}
}

SymTen1DyArray(d::AbstractArray{T,N}) where {T,N} =
    SymTenArray(ZeroArray(size(d)), ZeroArray(size(d)), ZeroArray(size(d)),
                                    d,                  ZeroArray(size(d)),
                                                        ZeroArray(size(d)))


const SymTen1DzArray{T, N} = SymTenArray{
    SymTen1Dz{T}, N, Array{Zero, N}, Array{Zero, N}, Array{Zero, N},
                                     Array{Zero, N}, Array{Zero, N},
                                                     Array{T, N}
}

SymTen1DzArray(f::AbstractArray{T,N}) where {T,N} =
    SymTenArray(ZeroArray(size(f)), ZeroArray(size(f)), ZeroArray(size(f)),
                                    ZeroArray(size(f)), ZeroArray(size(f)),
                                                        f)


const SymTenMaybe2DxyArray{T, Tz, N} = SymTenArray{
    SymTenMaybe2Dxy{T, Tz}, N, Array{T, N}, Array{T, N}, Array{Tz, N},
                                            Array{T, N}, Array{Tz, N},
                                                         Array{Tz, N}
}

@inline Base.size(A::SymTenArray) = size(A.xx)
@inline Base.length(A::SymTenArray) = length(A.xx)
Base.dataids(A::SymTenArray) = (Base.dataids(A.xx)..., Base.dataids(A.xy)..., Base.dataids(A.xz)...,
                                                       Base.dataids(A.yy)..., Base.dataids(A.yz)...,
                                                                              Base.dataids(A.zz)...)


@inline function Base.getindex(A::SymTenArray, i::Int)
    @boundscheck checkbounds(A, i)
    @inbounds r = SymTen(
        A.xx[i], A.xy[i], A.xz[i],
                 A.yy[i], A.yz[i],
                          A.zz[i]
    )
    return r
end

@inline function Base.getindex(A::SymTenArray{T, N}, I::Vararg{Int, N}) where {T, N}
    @boundscheck checkbounds(A, I...)
    @inbounds r = SymTen(
        A.xx[I...], A.xy[I...], A.xz[I...],
                    A.yy[I...], A.yz[I...],
                                A.zz[I...]
    )
    return r
end

@inline function Base.setindex!(A::SymTenArray{T}, s::AbstractTen, i::Int) where {T}
    @boundscheck checkbounds(A, i)

    sym_s = convert(T, s)
    @inbounds begin
        A.xx[i] = sym_s.xx
        A.xy[i] = sym_s.xy
        A.xz[i] = sym_s.xz
        A.yy[i] = sym_s.yy
        A.yz[i] = sym_s.yz
        A.zz[i] = sym_s.zz
    end

    return A
end

@inline function Base.setindex!(A::SymTenArray{T, N}, s::AbstractTen, I::Vararg{Int, N}) where {T, N}
    @boundscheck checkbounds(A, I...)

    sym_s = convert(T, s)
    @inbounds begin
        A.xx[I...] = sym_s.xx
        A.xy[I...] = sym_s.xy
        A.xz[I...] = sym_s.xz
        A.yy[I...] = sym_s.yy
        A.yz[I...] = sym_s.yz
        A.zz[I...] = sym_s.zz
    end

    return A
end

Base.similar(A::SymTenArray, ::Type{SymTen{Tt, Txx, Tyx, Tzx, Tyy, Tzy, Tzz}}, dims::Tuple{Int, Vararg{Int, N2}}) where {Tt, Txx, Tyx, Tzx, Tyy, Tzy, Tzz, N2} = SymTenArray(similar(A.xx, Txx, dims), similar(A.xy, Tyx, dims), similar(A.xz, Tzx, dims), similar(A.yy, Tyy, dims), similar(A.yz, Tzy, dims), similar(A.zz, Tzz, dims))

@inline function Base.getproperty(S::SymTenArray, s::Symbol)
    if s === :x
        xx = getfield(S, :xx)
        yx = getfield(S, :xy)
        zx = getfield(S, :xz)
        return VecArray(xx, yx, zx)
    elseif s === :y
        xy = getfield(S, :xy)
        yy = getfield(S, :yy)
        zy = getfield(S, :yz)
        return VecArray(xy, yy, zy)
    elseif s === :z
        xz = getfield(S, :xz)
        yz = getfield(S, :yz)
        zz = getfield(S, :zz)
        return VecArray(xz, yz, zz)
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

function Base.similar(bc::Broadcast.Broadcasted, ::Type{SymTen{T, Txx, Txy, Txz, Tyy, Tyz, Tzz}}) where {T, Txx, Txy, Txz, Tyy, Tyz, Tzz}
    s = length.(axes(bc))
    xx = Array{Txx}(undef, s...)
    xy = Array{Txy}(undef, s...)
    xz = Array{Txz}(undef, s...)
    yy = Array{Tyy}(undef, s...)
    yz = Array{Tyz}(undef, s...)
    zz = Array{Tzz}(undef, s...)
    return SymTenArray(xx, xy, xz, yy, yz, zz)
end

############################ AntiSymTenArray ###########################

struct AntiSymTenArray{T, N, Tyx, Tzx, Tzy} <: AbstractArray{T, N}
    xy::Tyx
    xz::Tzx
    yz::Tzy

    AntiSymTenArray{T}(I::Vararg{Int, N}) where {T, N} = new{AntiSymTen{T, T, T, T}, N, Array{T, N}, Array{T, N}, Array{T, N}}(
        Array{T}(undef, I...), Array{T}(undef, I...), Array{T}(undef, I...)
    )

    function AntiSymTenArray(xy::AbstractArray, xz::AbstractArray, yz::AbstractArray)

        s = size(xy)
        N = ndims(xy)
        size(xz) === s || throw(DimensionMismatch("Input Arrays must have the same size"))
        size(yz) === s || throw(DimensionMismatch("Input Arrays must have the same size"))


        Tyx = eltype(xy)
        Tzx = eltype(xz)
        Tzy = eltype(yz)
        Tf = promote_type_ignoring_Zero(Tyx, Tzx, Tzy)


        Tyxf = _my_promote_type(Tf, Tyx)
        Tzxf = _my_promote_type(Tf, Tzx)
        Tzyf = _my_promote_type(Tf, Tzy)
        Tff = _final_type(Tyxf, Tzxf, Tzyf)

        return new{
            AntiSymTen{Tff, Tyxf, Tzxf, Tzyf}, N,
            typeof(xy), typeof(xz), typeof(yz)
        }(xy, xz, yz)
    end

end

function AntiSymTenArray(xy, xz, yz)

    vals = (xy, xz, yz)

    vals === (𝟎, 𝟎, 𝟎) && throw(DomainError(vals, "At least one entry must be a valid Array"))

    non_zero_vals = _filter_zeros(vals...)
    s = size(non_zero_vals[1])

    all(x -> (size(x) === s), non_zero_vals) || throw(DimensionMismatch("Input Arrays must have the same size"))

    sizes = (s, s, s)
    final_vals = map(_if_zero_to_Array, sizes, vals)

    return AntiSymTenArray(final_vals...)
end

AntiSymTenArray(;xy::Union{Zero, <:AbstractArray} = 𝟎, xz::Union{Zero, <:AbstractArray} = 𝟎,
             yz::Union{Zero, <:AbstractArray} = 𝟎) = AntiSymTenArray(xy, xz, yz)


const AntiSymTen3DArray{T, N} = AntiSymTenArray{AntiSymTen3D{T}, N,
                                                Array{T, N}, Array{T, N}, Array{T, N}}

AntiSymTen3DArray(a::AbstractArray{T,N}, b::AbstractArray{T,N}, c::AbstractArray{T,N}) where {T,N} =
    AntiSymTenArray(a, b, c)


const AntiSymTen2DxyArray{T, N} = AntiSymTenArray{AntiSymTen2Dxy{T}, N,
                                                  Array{T, N}, Array{Zero, N}, Array{Zero, N}}

AntiSymTen2DxyArray(a::AbstractArray{T,N}) where {T,N} =
    AntiSymTenArray(a, ZeroArray(size(a)), ZeroArray(size(a)))


const AntiSymTen2DxzArray{T, N} = AntiSymTenArray{AntiSymTen2Dxz{T}, N,
                                                  Array{Zero, N}, Array{T, N}, Array{Zero, N}}

AntiSymTen2DxzArray(b::AbstractArray{T,N}) where {T,N} =
    AntiSymTenArray(ZeroArray(size(b)), b, ZeroArray(size(b)))


const AntiSymTen2DyzArray{T, N} = AntiSymTenArray{AntiSymTen2Dyz{T}, N,
                                                  Array{Zero, N}, Array{Zero, N}, Array{T, N}}

AntiSymTen2DyzArray(c::AbstractArray{T,N}) where {T,N} =
    AntiSymTenArray(ZeroArray(size(c)), ZeroArray(size(c)), c)


@inline Base.size(A::AntiSymTenArray) = size(A.xy)
@inline Base.length(A::AntiSymTenArray) = length(A.xy)
Base.dataids(A::AntiSymTenArray) = (Base.dataids(A.xy)..., Base.dataids(A.xz)..., Base.dataids(A.yz)...)

@inline function Base.getindex(A::AntiSymTenArray, i::Int)
    @boundscheck checkbounds(A, i)
    @inbounds r = AntiSymTen(A.xy[i], A.xz[i], A.yz[i])
    return r
end

@inline function Base.getindex(A::AntiSymTenArray{T, N}, I::Vararg{Int, N}) where {T, N}
    @boundscheck checkbounds(A, I...)
    @inbounds r = AntiSymTen(A.xy[I...], A.xz[I...], A.yz[I...])
    return r
end

@inline function Base.setindex!(A::AntiSymTenArray{T}, s::AbstractTen, i::Int) where {T}
    @boundscheck checkbounds(A, i)

    sym_s = convert(T, s)
    @inbounds begin
        A.xy[i] = sym_s.xy
        A.xz[i] = sym_s.xz
        A.yz[i] = sym_s.yz
    end

    return A
end

@inline function Base.setindex!(A::AntiSymTenArray{T, N}, s::AbstractTen, I::Vararg{Int, N}) where {T, N}
    @boundscheck checkbounds(A, I...)

    sym_s = convert(T, s)
    @inbounds begin
        A.xy[I...] = sym_s.xy
        A.xz[I...] = sym_s.xz
        A.yz[I...] = sym_s.yz
    end

    return A
end

Base.similar(A::AntiSymTenArray, ::Type{AntiSymTen{Tt, Tyx, Tzx, Tzy}}, dims::Tuple{Int, Vararg{Int, N2}}) where {Tt, Tyx, Tzx, Tzy, N2} = AntiSymTenArray(similar(A.xy, Tyx, dims), similar(A.xz, Tzx, dims), similar(A.yz, Tzy, dims))

@inline function Base.getproperty(S::AntiSymTenArray, s::Symbol)
    if s === :x
        yx = -getfield(S, :xy)
        zx = -getfield(S, :xz)
        return Vec2DyzArray(yx, zx)
    elseif s === :y
        xy = getfield(S, :xy)
        zy = -getfield(S, :yz)
        return Vec2DxzArray(xy, zy)
    elseif s === :z
        xz = getfield(S, :xz)
        yz = getfield(S, :yz)
        return Vec2DxyArray(xz, yz)
    elseif s === :yx
        return -getfield(S, :xy)
    elseif s === :zx
        return -getfield(S, :xz)
    elseif s === :zy
        return -getfield(S, :yz)
    elseif s === :xx
        return Array{Zero}(undef, size(S))
    elseif s === :yy
        return Array{Zero}(undef, size(S))
    elseif s === :zz
        return Array{Zero}(undef, size(S))
    else
        return getfield(S, s)
    end
end

function Base.similar(bc::Broadcast.Broadcasted, ::Type{AntiSymTen{T, Txy, Txz, Tyz}}) where {T, Txy, Txz, Tyz}
    s = length.(axes(bc))
    xy = Array{Txy}(undef, s...)
    xz = Array{Txz}(undef, s...)
    yz = Array{Tyz}(undef, s...)
    return AntiSymTenArray(xy, xz, yz)
end
