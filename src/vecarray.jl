struct VecArray{T,N,Tx,Ty,Tz} <: AbstractArray{T,N}
    x::Tx
    y::Ty
    z::Tz

    VecArray{T}(I::Vararg{Int,N}) where {T,N} = new{Vec{T,1,T,T,T},N,Array{T,N},Array{T,N},Array{T,N}}(Array{T}(undef,I...),Array{T}(undef,I...),Array{T}(undef,I...))
    
    function VecArray(x::AbstractArray{<:Number},y::AbstractArray{<:Number}, z::AbstractArray{<:Number})

        s = size(x)
        size(y) === s || throw(DomainError((x,y),"Arrays must have the same size"))
        size(z) === s || throw(DomainError((x,z),"Arrays must have the same size"))

        Tx = typeof(x)
        Ty = typeof(y)
        Tz = typeof(z)
        N = ndims(x)

        return new{_vec_type(eltype(x),eltype(y),eltype(z)),N,Tx,Ty,Tz}(x,y,z)
    end

    function VecArray(x::AbstractArray{<:AbstractVec},y::AbstractArray{<:AbstractVec}, z::AbstractArray{<:AbstractVec})

        s = size(x)
        size(y) === s || throw(DomainError((x,y),"Arrays must have the same size"))
        size(z) === s || throw(DomainError((x,z),"Arrays must have the same size"))

        Tx = eltype(x)
        Ty = eltype(y)
        Tz = eltype(z)

        N = ndims(x)

        return new{_ten_type(Tx,Ty,Tz),N,typeof(x),typeof(y),typeof(z)}(x,y,z)
    end

end

function VecArray(;x::Union{Zero,<:AbstractArray{<:Number}}=𝟎,y::Union{Zero,AbstractArray{<:Number}}=𝟎, z::Union{Zero,AbstractArray{<:Number}}=𝟎)
    if (x,y,z) === (𝟎,𝟎,𝟎)
        throw(DomainError((x,y,z),"At least one entry must be a valid array"))
    else
        if x !== 𝟎
            s = size(x)
            xv = x
            yv = y === 𝟎 ? Array{Zero}(undef,s) : y
            zv = z === 𝟎 ? Array{Zero}(undef,s) : z
            size(yv) === s || throw(DomainError((x,y),"Arrays must have the same size"))
            size(zv) === s || throw(DomainError((x,z),"Arrays must have the same size"))
        elseif y !== 𝟎
            s = size(y)
            xv = Array{Zero}(undef,s)
            yv = y
            zv = z === 𝟎 ? Array{Zero}(undef,s) : z
            size(zv) === s || throw(DomainError((y,z),"Arrays must have the same size"))
        else # z must be !== 𝟎
            s = size(z)
            xv = Array{Zero}(undef,s)
            yv = Array{Zero}(undef,s)
            zv = z 
        end
    end
    return VecArray(xv,yv,zv)
end

_if_zero_to_Array(s::NTuple{N,Int},::Zero) where N = Array{Zero}(undef,s)
_if_zero_to_Array(s::NTuple{N,Int},x::AbstractArray) where N = x

function TenArray(xx, yx, zx,
                  xy, yy, zy,
                  xz, yz, zz)

    vals = (xx,yx,zx,
            xy,yy,zy,
            xz,yz,zz)

    vals === (𝟎,𝟎,𝟎,
              𝟎,𝟎,𝟎,
              𝟎,𝟎,𝟎) && throw(DomainError( vals,"At least one entry must be a valid Array"))

    non_zero_vals = _filter_zeros(vals...)
    s = size(non_zero_vals[1])

    all(x->(size(x)===s),non_zero_vals) || throw(DimensionMismatch())

    sizes = (s,s,s,s,s,s,s,s,s)
    final_vals = map(_if_zero_to_Array,sizes,vals)

    xv = VecArray(final_vals[1],final_vals[2],final_vals[3])
    yv = VecArray(final_vals[4],final_vals[5],final_vals[6])
    zv = VecArray(final_vals[7],final_vals[8],final_vals[9])

    return VecArray(xv,yv,zv)
end
 
TenArray(;xx::Union{Zero,<:AbstractArray{<:Number}}=𝟎, yx::Union{Zero,<:AbstractArray{<:Number}}=𝟎, zx::Union{Zero,<:AbstractArray{<:Number}}=𝟎,
          xy::Union{Zero,<:AbstractArray{<:Number}}=𝟎, yy::Union{Zero,<:AbstractArray{<:Number}}=𝟎, zy::Union{Zero,<:AbstractArray{<:Number}}=𝟎,
          xz::Union{Zero,<:AbstractArray{<:Number}}=𝟎, yz::Union{Zero,<:AbstractArray{<:Number}}=𝟎, zz::Union{Zero,<:AbstractArray{<:Number}}=𝟎) = TenArray(xx,yx,zx,
                                                                                                                                                            xy,yy,zy,
                                                                                                                                                            xz,yz,zz)

const Vec3DArray{T,N} = VecArray{Vec3D{T},N,Array{T,N},Array{T,N},Array{T,N}}
const Vec2DxyArray{T,N} = VecArray{Vec2Dxy{T},N,Array{T,N},Array{T,N},Array{Zero,N}}
const Vec2DxzArray{T,N} = VecArray{Vec2Dxz{T},N,Array{T,N},Array{Zero,N},Array{T,N}}
const Vec2DyzArray{T,N} = VecArray{Vec2Dyz{T},N,Array{Zero,N},Array{T,N},Array{T,N}}
const Vec2DArray{T,N} = Union{Vec2DxyArray{T,N},Vec2DxzArray{T,N},Vec2DyzArray{T,N}}
const Vec1DxArray{T,N} = VecArray{Vec1Dx{T},N,Array{T,N},Array{Zero,N},Array{Zero,N}}
const Vec1DyArray{T,N} = VecArray{Vec1Dy{T},N,Array{Zero,N},Array{T,N},Array{Zero,N}}
const Vec1DzArray{T,N} = VecArray{Vec1Dz{T},N,Array{Zero,N},Array{Zero,N},Array{T,N}}
const Vec1DArray{T,N} = Union{Vec1DxArray{T,N},Vec1DyArray{T,N},Vec1DzArray{T,N}}
const Vec0DArray{N} = VecArray{Vec0D,N,Array{Zero,N},Array{Zero,N},Array{Zero,N}}
const VecNDArray{T,N} = Union{Vec3DArray{T,N},Vec2DArray{T,N},Vec1DArray{T,N}}

const VecMaybe2DxyArray{T,Tz,N} = VecArray{VecMaybe2Dxy{T,Tz},N,Array{T,N},Array{T,N},Array{Tz,N}}

const Ten3DArray{T,N} = VecArray{Ten3D{T},N,Vec3DArray{T,N},Vec3DArray{T,N},Vec3DArray{T,N}}
const Ten2DxyArray{T,N} = VecArray{Ten2Dxy{T},N,Vec2DxyArray{T,N},Vec2DxyArray{T,N},Vec0DArray{N}}
const Ten2DxzArray{T,N} = VecArray{Ten2Dxz{T},N,Vec2DxzArray{T,N},Vec0DArray{N},Vec2DxzArray{T,N}}
const Ten2DyzArray{T,N} = VecArray{Ten2Dyz{T},N,Vec0DArray{N},Vec2DyzArray{T,N},Vec2DyzArray{T,N}}
const Ten2DArray{T,N} = Union{Ten2DxyArray{T,N},Ten2DxzArray{T,N},Ten2DyzArray{T,N}}
const Ten1DxArray{T,N} = VecArray{Ten1Dx{T},N,Vec1DxArray{T,N},Vec0DArray{N},Vec0DArray{N}}
const Ten1DyArray{T,N} = VecArray{Ten1Dy{T},N,Vec0DArray{N},Vec1DyArray{T,N},Vec0DArray{N}}
const Ten1DzArray{T,N} = VecArray{Ten1Dz{T},N,Vec0DArray{N},Vec0DArray{N},Vec1DzArray{T,N}}
const Ten1DArray{T,N} = Union{Ten1DxArray{T,N},Ten1DyArray{T,N},Ten1DzArray{T,N}}
const TenNDArray{T,N} = Union{Ten3DArray{T,N},Ten2DArray{T,N},Ten1DArray{T,N}}

const TenMaybe2DxyArray{T,Tz,N} = VecArray{TenMaybe2Dxy{T,Tz},N,VecMaybe2DxyArray{T,Tz,N},VecMaybe2DxyArray{T,Tz,N},Vec3DArray{Tz,N}}

@inline Base.size(A::VecArray) = size(A.x)
@inline Base.length(A::VecArray) = length(A.x)

@inline function Base.getindex(A::VecArray,i::Int)
    @boundscheck checkbounds(A,i)
    @inbounds r = Vec(A.x[i], A.y[i], A.z[i])
    return r
end

@inline function Base.getindex(A::VecArray{T,N},I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A,I...)
    @inbounds r = Vec(A.x[I...], A.y[I...], A.z[I...])
    return r
end

@inline function Base.setindex!(A::VecArray, u::AbstractVec, i::Int)
    @boundscheck checkbounds(A,i)

    @inbounds begin 
        A.x[i] = u.x
        A.y[i] = u.y
        A.z[i] = u.z
    end

    return A
end

@inline function Base.setindex!(A::VecArray{T,N}, u::AbstractVec, I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A,I...)

    @inbounds begin 
        A.x[I...] = u.x
        A.y[I...] = u.y
        A.z[I...] = u.z
    end

    return A
end

Base.similar(A::VecArray,T::Type{Vec{Tt,N,Tx,Ty,Tz}},dims::Tuple{Int,Vararg{Int,N2}}) where {Tt,N,Tx,Ty,Tz,N2} = VecArray(similar(A.x,Tx,dims), similar(A.y,Ty,dims), similar(A.z,Tz,dims))

#Definitons so broadcast return a VecArray =======================================

function Base.similar(bc::Broadcast.Broadcasted, ::Type{Vec{T,1,Tx,Ty,Tz}}) where {T,Tx,Ty,Tz}
    s = length.(axes(bc))
    x = Array{Tx}(undef,s...)
    y = Array{Ty}(undef,s...)
    z = Array{Tz}(undef,s...)
    return VecArray(x,y,z)
end

function Base.similar(bc::Broadcast.Broadcasted, ::Type{Vec{T,2,Tx,Ty,Tz}}) where {T,Tx,Ty,Tz}
    xv = similar(bc,Tx)
    yv = similar(bc,Ty)
    zv = similar(bc,Tz)
    return VecArray(xv,yv,zv)
end

#Definitons so broadcast return a VecArray =======================================

@inline function Base.getproperty(T::TenNDArray,s::Symbol)
    if s === :xx
        return getfield(getfield(T,:x),:x)
    elseif s === :xy
        return getfield(getfield(T,:y),:x)
    elseif s === :xz
        return getfield(getfield(T,:z),:x)
    elseif s === :yx
        return getfield(getfield(T,:x),:y)
    elseif s === :yy
        return getfield(getfield(T,:y),:y)
    elseif s === :yz
        return getfield(getfield(T,:z),:y)
    elseif s === :zx
        return getfield(getfield(T,:x),:z)
    elseif s === :zy
        return getfield(getfield(T,:y),:z)
    elseif s === :zz
        return getfield(getfield(T,:z),:z)
    else
        return getfield(T,s)
    end
end
