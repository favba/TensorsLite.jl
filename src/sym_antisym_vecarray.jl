struct SymTenArray{T,N,Txx,Tyx,Tzx,Tyy,Tzy,Tzz} <: AbstractArray{T,N}
    xx::Txx
    yx::Tyx
    zx::Tzx
    yy::Tyy
    zy::Tzy
    zz::Tzz

    SymTenArray{T}(I::Vararg{Int,N}) where {T,N} = new{SymTen{T,T,T,T,T,T,T},N,Array{T,N},Array{T,N},Array{T,N},Array{T,N},Array{T,N},Array{T,N}}(
                              Array{T}(undef,I...),Array{T}(undef,I...),Array{T}(undef,I...),
                              Array{T}(undef,I...),Array{T}(undef,I...),Array{T}(undef,I...))

    function SymTenArray(xx::AbstractArray{<:Number},yx::AbstractArray{<:Number},zx::AbstractArray{<:Number},
                                                     yy::AbstractArray{<:Number},zy::AbstractArray{<:Number},
                                                                                 zz::AbstractArray{<:Number})

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
        Tf = promote_type(Txx,Tyx,Tzx,
                              Tyy,Tzy,
                                  Tzz)


        Txxf = _my_promote_type(Tf,Txx)
        Tyxf = _my_promote_type(Tf,Tyx)
        Tzxf = _my_promote_type(Tf,Tzx)
        Tyyf = _my_promote_type(Tf,Tyy)
        Tzyf = _my_promote_type(Tf,Tzy)
        Tzzf = _my_promote_type(Tf,Tzz)
        Tff = _final_type(Txxf,Tyxf,Tzxf,Tyyf,Tzyf,Tzzf)

        return new{SymTen{Tff,Txxf,Tyxf,Tzxf,Tyyf,Tzyf,Tzzf},N,
                   typeof(xx),typeof(yx),typeof(zx),
                              typeof(yy),typeof(zy),
                                         typeof(zz)}(xx, yx, zx,
                                                         yy, zy,
                                                             zz)
    end

end

function SymTenArray(xx, yx, zx,
                         yy, zy,
                             zz)

    vals = (xx,yx,zx,
               yy,zy,
                  zz)

    vals === (𝟎,𝟎,𝟎,
                𝟎,𝟎,
                  𝟎) && throw(DomainError( vals,"At least one entry must be a valid Array"))

    non_zero_vals = _filter_zeros(vals...)
    s = size(non_zero_vals[1])

    all(x->(size(x)===s),non_zero_vals) || throw(DimensionMismatch("Input Arrays must have the same size"))

    sizes = (s,s,s,s,s,s)
    final_vals = map(_if_zero_to_Array,sizes,vals)

    return SymTenArray(final_vals...)
end
 
SymTenArray(;xx::Union{Zero,<:AbstractArray{<:Number}}=𝟎, yx::Union{Zero,<:AbstractArray{<:Number}}=𝟎, zx::Union{Zero,<:AbstractArray{<:Number}}=𝟎,
             yy::Union{Zero,<:AbstractArray{<:Number}}=𝟎, zy::Union{Zero,<:AbstractArray{<:Number}}=𝟎,
             zz::Union{Zero,<:AbstractArray{<:Number}}=𝟎) = SymTenArray(xx, yx, zx,
                                                                            yy, zy,
                                                                                zz)

const SymTen3DArray{T,N} = SymTenArray{SymTen3D{T},N,Array{T,N},Array{T,N},Array{T,N},
                                                                Array{T,N},Array{T,N},
                                                                           Array{T,N}}
const SymTen2DxyArray{T,N} = SymTenArray{SymTen2Dxy{T},N,Array{T,N},Array{T,N},Array{Zero,N},
                                                                    Array{T,N},Array{Zero,N},
                                                                               Array{Zero,N}}
const SymTen2DxzArray{T,N} = SymTenArray{SymTen2Dxz{T},N,Array{T,N},Array{Zero,N},Array{T,N},
                                                                    Array{Zero,N},Array{Zero,N},
                                                                                 Array{T,N}}
const SymTen2DyzArray{T,N} = SymTenArray{SymTen2Dyz{T},N,Array{Zero,N},Array{Zero,N},Array{Zero,N},
                                                                      Array{T,N},  Array{T,N},
                                                                                   Array{T,N}}
const SymTen1DxArray{T,N} = SymTenArray{SymTen1Dx{T},N,Array{T,N},Array{Zero,N},Array{Zero,N},
                                                                  Array{Zero,N},Array{Zero,N},
                                                                               Array{Zero,N}}
const SymTen1DyArray{T,N} = SymTenArray{SymTen1Dx{T},N,Array{Zero,N},Array{Zero,N},Array{Zero,N},
                                                                    Array{T,N},Array{Zero,N},
                                                                               Array{Zero,N}}
const SymTen1DzArray{T,N} = SymTenArray{SymTen1Dx{T},N,Array{Zero,N},Array{Zero,N},Array{Zero,N},
                                                                    Array{Zero,N},Array{Zero,N},
                                                                                 Array{T,N}}

@inline Base.size(A::SymTenArray) = size(A.xx)
@inline Base.length(A::SymTenArray) = length(A.xx)

@inline function Base.getindex(A::SymTenArray,i::Int)
    @boundscheck checkbounds(A,i)
    @inbounds r = SymTen(A.xx[i], A.yx[i], A.zx[i],
                                  A.yy[i], A.zy[i],
                                           A.zz[i])
    return r
end

@inline function Base.getindex(A::SymTenArray{T,N},I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A,I...)
    @inbounds r = SymTen(A.xx[I...], A.yx[I...], A.zx[I...],
                                     A.yy[I...], A.zy[I...],
                                                 A.zz[I...])
    return r
end

@inline function Base.setindex!(A::SymTenArray{T}, s::AbstractTen, i::Int) where T
    @boundscheck checkbounds(A,i)

    sym_s = convert(T,s)
    @inbounds begin 
        A.xx[i] = sym_s.xx
        A.yx[i] = sym_s.yx
        A.zx[i] = sym_s.zx
        A.yy[i] = sym_s.yy
        A.zy[i] = sym_s.zy
        A.zz[i] = sym_s.zz
    end

    return A
end

@inline function Base.setindex!(A::SymTenArray{T,N}, s::AbstractTen, I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A,I...)

    sym_s = convert(T,s)
    @inbounds begin 
        A.xx[I...] = sym_s.xx
        A.yx[I...] = sym_s.yx
        A.zx[I...] = sym_s.zx
        A.yy[I...] = sym_s.yy
        A.zy[I...] = sym_s.zy
        A.zz[I...] = sym_s.zz
    end

    return A
end

Base.similar(A::SymTenArray,T::Type{SymTen{Tt,N,Tx,Ty,Tz}},dims::Tuple{Int,Vararg{Int,N2}}) where {Tt,N,Tx,Ty,Tz,N2} = SymTenArray(similar(A.x,Tx,dims), similar(A.y,Ty,dims), similar(A.z,Tz,dims))

@inline function Base.getproperty(S::SymTenArray,s::Symbol)
    if s === :x
        xx = getfield(S,:xx)
        yx = getfield(S,:yx)
        zx = getfield(S,:zx)
        return VecArray(xx,yx,zx)
    elseif s === :y
        xy = getfield(S,:yx)
        yy = getfield(S,:yy)
        zy = getfield(S,:zy)
        return VecArray(xy,yy,zy)
    elseif s === :z
        xz = getfield(S,:zx)
        yz = getfield(S,:zy)
        zz = getfield(S,:zz)
        return VecArray(xz,yz,zz)
    else
        return getfield(S,s)
    end
end

function Base.similar(bc::Broadcast.Broadcasted, ::Type{SymTen{T,Txx,Txy,Txz,Tyy,Tyz,Tzz}}) where {T,Txx,Txy,Txz,Tyy,Tyz,Tzz}
    s = length.(axes(bc))
    xx = Array{Txx}(undef,s...)
    xy = Array{Txy}(undef,s...)
    xz = Array{Txz}(undef,s...)
    yy = Array{Tyy}(undef,s...)
    yz = Array{Tyz}(undef,s...)
    zz = Array{Tzz}(undef,s...)
    return SymTenArray(xx,xy,xz,yy,yz,zz)
end
