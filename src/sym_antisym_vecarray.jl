const SymTenArray{T,N,Txx,Txy,Txz,Tyy,Tyz,Tzz} =
    AbstractTensorArray{SymmetricTensor{2,T,Txx,Txy,Txz,Tyy,Tyz,Tzz},N}

struct SymmetricTensorArray{T, N, Txx, Tyx, Tzx, Tyy, Tzy, Tzz} <: AbstractTensorArray{T, N}
    xx::Txx
    xy::Tyx
    xz::Tzx
    yy::Tyy
    yz::Tzy
    zz::Tzz

    function SymmetricTensorArray(
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
        Tf = promote_type_ignoring_Zero_and_One(
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
        Tff = Union{Txxf, Tyxf, Tzxf, Tyyf, Tzyf, Tzzf}

        return new{
            SymmetricTensor{2, Tff, Txxf, Tyxf, Tzxf, Tyyf, Tzyf, Tzzf}, N,
            typeof(xx), typeof(yx), typeof(zx),
                        typeof(yy), typeof(zy),
                                    typeof(zz)
        }(
            xx, yx, zx,
                yy, zy,
                    zz
        )
    end

    function SymmetricTensorArray(xx::AbstractArray{Txx, N}, xy::AbstractArray{Txy, N}, xz::AbstractArray{Txz, N},
                                  yy::AbstractArray{Tyy, N}, yz::AbstractArray{Tyz, N},
                                  zz::AbstractArray{Tzz, N}) where {NV,
                                                                    Txx<:AbstractTensor{NV},
                                                                    Txy<:AbstractTensor{NV},
                                                                    Txz<:AbstractTensor{NV},
                                                                    Tyy<:AbstractTensor{NV},
                                                                    Tyz<:AbstractTensor{NV},
                                                                    Tzz<:AbstractTensor{NV},
                                                                    N}

        s = size(xx)
        (size(xy) === s && size(xz) === s && size(yy) === s && size(yz) === s && size(zz) === S) ||
            throw(DimensionMismatch("Arrays must have the same size"))

        # Figure out how to not rely on promote_op
        TT = Base.promote_op(SymmetricTensor,Txx,Txy,Txz,Tyy,Tyz,Tzz)

        return new{TT, N,
                   typeof(xx), typeof(xy), typeof(xz),
                   typeof(yy), typeof(yz),
                   typeof(zz)}(xx, xy, xz,
                               yy, yz,
                               zz)
    end
end

function SymmetricTensorArray(
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

    return SymmetricTensorArray(final_vals...)
end

########### Aliases #################

const SymTen3DArray{T, N} = SymmetricTensorArray{
    SymTen3D{T}, N, Array{T, N}, Array{T, N}, Array{T, N},
                                 Array{T, N}, Array{T, N},
                                              Array{T, N}
}


const SymTen2DxyArray{T, N} = SymmetricTensorArray{
    SymTen2Dxy{T}, N, Array{T, N}, Array{T, N}, Array{Zero, N},
                                   Array{T, N}, Array{Zero, N},
                                                Array{Zero, N}
}

const SymTen2DxzArray{T, N} = SymmetricTensorArray{
    SymTen2Dxz{T}, N, Array{T, N}, Array{Zero, N}, Array{T, N},
                                   Array{Zero, N}, Array{Zero, N},
                                                   Array{T, N}
}

const SymTen2DyzArray{T, N} = SymmetricTensorArray{
    SymTen2Dyz{T}, N, Array{Zero, N}, Array{Zero, N}, Array{Zero, N},
                                      Array{T, N},    Array{T, N},
                                                      Array{T, N}
}

const SymTen1DxArray{T, N} = SymmetricTensorArray{
    SymTen1Dx{T}, N, Array{T, N}, Array{Zero, N}, Array{Zero, N},
                                  Array{Zero, N}, Array{Zero, N},
                                                  Array{Zero, N}
}

const SymTen1DyArray{T, N} = SymmetricTensorArray{
    SymTen1Dy{T}, N, Array{Zero, N}, Array{Zero, N}, Array{Zero, N},
                                     Array{T, N},    Array{Zero, N},
                                                     Array{Zero, N}
}

const SymTen1DzArray{T, N} = SymmetricTensorArray{
    SymTen1Dz{T}, N, Array{Zero, N}, Array{Zero, N}, Array{Zero, N},
                                     Array{Zero, N}, Array{Zero, N},
                                                     Array{T, N}
}

const SymTenMaybe2DxyArray{T, Tz, N} = SymmetricTensorArray{
    SymTenMaybe2Dxy{T, Tz}, N, Array{T, N}, Array{T, N}, Array{Tz, N},
                                            Array{T, N}, Array{Tz, N},
                                                         Array{Tz, N}
}

############ Constructors ###################

SymmetricTensorArray(;
    xx::Union{Zero, <:AbstractArray} = 𝟎, xy::Union{Zero, <:AbstractArray} = 𝟎, xz::Union{Zero, <:AbstractArray} = 𝟎,
                                          yy::Union{Zero, <:AbstractArray} = 𝟎, yz::Union{Zero, <:AbstractArray} = 𝟎,
                                                                                zz::Union{Zero, <:AbstractArray} = 𝟎
) = SymmetricTensorArray(
    xx, xy, xz,
        yy, yz,
            zz
)

SymTen3DArray(a::AbstractArray{T,N}, b::AbstractArray{T,N}, c::AbstractArray{T,N},
                                     d::AbstractArray{T,N}, e::AbstractArray{T,N},
                                                            f::AbstractArray{T,N}) where {T,N} =
    SymmetricTensorArray(a, b, c,
                   d, e,
                      f)


SymTen2DxyArray(a::AbstractArray{T,N}, b::AbstractArray{T,N},
                d::AbstractArray{T,N}) where {T,N} =
    SymmetricTensorArray(a, b, ZeroArray(size(a)),
                   d, ZeroArray(size(a)),
                      ZeroArray(size(a)))


SymTen2DxzArray(a::AbstractArray{T,N}, c::AbstractArray{T,N},
                                       f::AbstractArray{T,N}) where {T,N} =
    SymmetricTensorArray(a, ZeroArray(size(a)), c,
                   ZeroArray(size(a)), ZeroArray(size(a)),
                                       f)


SymTen2DyzArray(d::AbstractArray{T,N}, e::AbstractArray{T,N},
                                       f::AbstractArray{T,N}) where {T,N} =
    SymmetricTensorArray(ZeroArray(size(d)), ZeroArray(size(d)), ZeroArray(size(d)),
                d,                  e,
                                                        f)


SymTen1DxArray(a::AbstractArray{T,N}) where {T,N} =
    SymmetricTensorArray(a, ZeroArray(size(a)), ZeroArray(size(a)),
                   ZeroArray(size(a)), ZeroArray(size(a)),
                                       ZeroArray(size(a)))


SymTen1DyArray(d::AbstractArray{T,N}) where {T,N} =
    SymmetricTensorArray(ZeroArray(size(d)), ZeroArray(size(d)), ZeroArray(size(d)),
                                    d,                  ZeroArray(size(d)),
                                                        ZeroArray(size(d)))


SymTen1DzArray(f::AbstractArray{T,N}) where {T,N} =
    SymmetricTensorArray(ZeroArray(size(f)), ZeroArray(size(f)), ZeroArray(size(f)),
                                    ZeroArray(size(f)), ZeroArray(size(f)),
                                                        f)


###### AbstractArray Interface #######

Base.dataids(A::SymmetricTensorArray) = (Base.dataids(A.xx)..., Base.dataids(A.xy)..., Base.dataids(A.xz)..., Base.dataids(A.yy)..., Base.dataids(A.yz)..., Base.dataids(A.zz)...)

Base.similar(A::SymmetricTensorArray, ::Type{SymmetricTensor{N, Tt, Txx, Tyx, Tzx, Tyy, Tzy, Tzz}}, dims::Tuple{Int, Vararg{Int, N2}}) where {N, Tt, Txx, Tyx, Tzx, Tyy, Tzy, Tzz, N2} = SymmetricTensorArray(similar(A.xx, Txx, dims), similar(A.xy, Tyx, dims), similar(A.xz, Tzx, dims), similar(A.yy, Tyy, dims), similar(A.yz, Tzy, dims), similar(A.zz, Tzz, dims))

@inline function Base.getproperty(S::SymmetricTensorArray, s::Symbol)
    if s === :x
        xx = getfield(S, :xx)
        yx = getfield(S, :xy)
        zx = getfield(S, :xz)
        return TensorArray(xx, yx, zx)
    elseif s === :y
        xy = getfield(S, :xy)
        yy = getfield(S, :yy)
        zy = getfield(S, :yz)
        return TensorArray(xy, yy, zy)
    elseif s === :z
        xz = getfield(S, :xz)
        yz = getfield(S, :yz)
        zz = getfield(S, :zz)
        return TensorArray(xz, yz, zz)
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

function Base.similar(bc::Broadcast.Broadcasted, ::Type{SymmetricTensor{2, T, Txx, Txy, Txz, Tyy, Tyz, Tzz}}) where {T, Txx, Txy, Txz, Tyy, Tyz, Tzz}
    s = length.(axes(bc))
    xx = Array{Txx}(undef, s...)
    xy = Array{Txy}(undef, s...)
    xz = Array{Txz}(undef, s...)
    yy = Array{Tyy}(undef, s...)
    yz = Array{Tyz}(undef, s...)
    zz = Array{Tzz}(undef, s...)
    return SymmetricTensorArray(xx, xy, xz, yy, yz, zz)
end

function Base.similar(bc::Broadcast.Broadcasted, ::Type{SymmetricTensor{N, T, Txx, Txy, Txz, Tyy, Tyz, Tzz}}) where {T, N, Txx, Txy, Txz, Tyy, Tyz, Tzz}

    xxv = similar(bc, Txx)
    xyv = similar(bc, Txy)
    xzv = similar(bc, Txz)
    yyv = similar(bc, Tyy)
    yzv = similar(bc, Tyz)
    zzv = similar(bc, Tzz)
    
    return SymmetricTensorArray(xxv, xyv, xzv, yyv, yzv, zzv)
end

############################ AntiSymTenArray ###########################

const AntiSymTenArray{T,N,Txy,Txz,Tyz} =
    AbstractTensorArray{AntiSymmetricTensor{2,Txy,Txz,Tyz},N}

struct AntiSymmetricTensorArray{T, N, Tyx, Tzx, Tzy} <: AbstractTensorArray{T, N}
    xy::Tyx
    xz::Tzx
    yz::Tzy

    function AntiSymmetricTensorArray(xy::AbstractArray, xz::AbstractArray, yz::AbstractArray)

        s = size(xy)
        N = ndims(xy)
        size(xz) === s || throw(DimensionMismatch("Input Arrays must have the same size"))
        size(yz) === s || throw(DimensionMismatch("Input Arrays must have the same size"))


        Tyx = eltype(xy)
        Tzx = eltype(xz)
        Tzy = eltype(yz)
        Tf = promote_type_ignoring_Zero_and_One(Tyx, Tzx, Tzy)


        Tyxf = _my_promote_type(Tf, Tyx)
        Tzxf = _my_promote_type(Tf, Tzx)
        Tzyf = _my_promote_type(Tf, Tzy)
        Tff = Union{Tyxf, Tzxf, Tzyf}

        return new{
            AntiSymmetricTensor{2, Union{Tff,Zero}, Tyxf, Tzxf, Tzyf}, N,
            typeof(xy), typeof(xz), typeof(yz)
        }(xy, xz, yz)
    end

    function AntiSymmetricTensorArray(xy::AbstractArray{Txy, N}, xz::AbstractArray{Txz, N}, yz::AbstractArray{Tyz, N}) where {N, NV, Txy<:AbstractTensor{NV}, Txz<:AbstractTensor{NV}, Tyz<:AbstractTensor{NV}}

        s = size(xy)
        size(xz) === s || throw(DimensionMismatch("Arrays must have the same size"))
        size(yz) === s || throw(DimensionMismatch("Arrays must have the same size"))

        # Figure out how to not rely on promote_op
        TT = Base.promote_op(AntiSymmetricTensor,Txy,Txz,Tyz)

        return new{TT, N, typeof(xy), typeof(xz), typeof(yz)}(xy, xz, yz)
    end
end

function AntiSymmetricTensorArray(xy, xz, yz)

    vals = (xy, xz, yz)

    vals === (𝟎, 𝟎, 𝟎) && throw(DomainError(vals, "At least one entry must be a valid Array"))

    non_zero_vals = _filter_zeros(vals...)
    s = size(non_zero_vals[1])

    all(x -> (size(x) === s), non_zero_vals) || throw(DimensionMismatch("Input Arrays must have the same size"))

    sizes = (s, s, s)
    final_vals = map(_if_zero_to_Array, sizes, vals)

    return AntiSymmetricTensorArray(final_vals...)
end

AntiSymmetricTensorArray(;xy::Union{Zero, <:AbstractArray} = 𝟎, xz::Union{Zero, <:AbstractArray} = 𝟎,
             yz::Union{Zero, <:AbstractArray} = 𝟎) = AntiSymmetricTensorArray(xy, xz, yz)


const AntiSymTen3DArray{T, N} = AntiSymmetricTensorArray{AntiSymTen3D{T}, N,
                                                Array{T, N}, Array{T, N}, Array{T, N}}


const AntiSymTen2DxyArray{T, N} = AntiSymmetricTensorArray{AntiSymTen2Dxy{T}, N,
                                                  Array{T, N}, Array{Zero, N}, Array{Zero, N}}


const AntiSymTen2DxzArray{T, N} = AntiSymmetricTensorArray{AntiSymTen2Dxz{T}, N,
                                                  Array{Zero, N}, Array{T, N}, Array{Zero, N}}


const AntiSymTen2DyzArray{T, N} = AntiSymmetricTensorArray{AntiSymTen2Dyz{T}, N,
                                                  Array{Zero, N}, Array{Zero, N}, Array{T, N}}

AntiSymTen2DyzArray(c::AbstractArray{T,N}) where {T,N} =
    AntiSymmetricTensorArray(ZeroArray(size(c)), ZeroArray(size(c)), c)

AntiSymTen3DArray(a::AbstractArray{T,N}, b::AbstractArray{T,N}, c::AbstractArray{T,N}) where {T,N} =
    AntiSymmetricTensorArray(a, b, c)

AntiSymTen2DxyArray(a::AbstractArray{T,N}) where {T,N} =
    AntiSymmetricTensorArray(a, ZeroArray(size(a)), ZeroArray(size(a)))

AntiSymTen2DxzArray(b::AbstractArray{T,N}) where {T,N} =
    AntiSymmetricTensorArray(ZeroArray(size(b)), b, ZeroArray(size(b)))


Base.dataids(A::AntiSymmetricTensorArray) = (Base.dataids(A.xy)..., Base.dataids(A.xz)..., Base.dataids(A.yz)...)

Base.similar(A::AntiSymmetricTensorArray, ::Type{AntiSymmetricTensor{N, Tt, Tyx, Tzx, Tzy}}, dims::Tuple{Int, Vararg{Int, N2}}) where {N, Tt, Tyx, Tzx, Tzy, N2} = AntiSymmetricTensorArray(similar(A.xy, Tyx, dims), similar(A.xz, Tzx, dims), similar(A.yz, Tzy, dims))

# @inline function Base.getproperty(S::AntiSymmetricTensorArray, s::Symbol)
#     if s === :x
#         yx = -getfield(S, :xy)
#         zx = -getfield(S, :xz)
#         return Tensor2DyzArray(yx, zx)
#     elseif s === :y
#         xy = getfield(S, :xy)
#         zy = -getfield(S, :yz)
#         return Tensor2DxzArray(xy, zy)
#     elseif s === :z
#         xz = getfield(S, :xz)
#         yz = getfield(S, :yz)
#         return Tensor2DxyArray(xz, yz)
#     elseif s === :yx
#         return -getfield(S, :xy)
#     elseif s === :zx
#         return -getfield(S, :xz)
#     elseif s === :zy
#         return -getfield(S, :yz)
    # elseif s === :xx
    #     return Array{Zero}(undef, size(S))
    # elseif s === :yy
    #     return Array{Zero}(undef, size(S))
    # elseif s === :zz
    #     return Array{Zero}(undef, size(S))
    # else
    #     return getfield(S, s)
#     end
# end

function Base.similar(bc::Broadcast.Broadcasted, ::Type{AntiSymmetricTensor{2, T, Txy, Txz, Tyz}}) where {T, Txy, Txz, Tyz}
    s = length.(axes(bc))
    xy = Array{Txy}(undef, s...)
    xz = Array{Txz}(undef, s...)
    yz = Array{Tyz}(undef, s...)
    return AntiSymmetricTensorArray(xy, xz, yz)
end

function Base.similar(bc::Broadcast.Broadcasted, ::Type{AntiSymmetricTensor{N, T, Txy, Txz, Tyz}}) where {T, N, Txy, Txz, Tyz}
    xyv = similar(bc, Txy)
    xzv = similar(bc, Txz)
    yzv = similar(bc, Tyz)
    return AntiSymmetricTensorArray(xyv, xzv, yzv)
end

