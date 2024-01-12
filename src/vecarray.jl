struct ZeroArray{N} <: AbstractArray{Zero,N}
    size::NTuple{N,Int}
end
ZeroArray(I::Vararg{Int,N}) where N = ZeroArray{N}((I...,))
Base.size(z::ZeroArray) = z.size
@inline Base.getindex(z::ZeroArray,i::Int) = 𝟎
@inline Base.getindex(z::ZeroArray{N},I::Vararg{Int,N}) where N = 𝟎
@inline Base.setindex!(z::ZeroArray,x,i::Int) = begin convert(Zero,x); return z end
@inline Base.setindex!(z::ZeroArray{N},x,I::Vararg{Int,N}) where N = begin convert(Zero,x); return z end

Base.similar(::ZeroArray,::Type{Zero},dims::Tuple{Int,Vararg{Int,N}}) where N = ZeroArray(dims...)

struct VecArray{T,N,Tx,Ty,Tz} <: AbstractArray{T,N}
    x::Tx
    y::Ty
    z::Tz

    VecArray{T}(I::Vararg{Int,N}) where {T,N} = new{Vec{T,1,T,T,T},N,Array{T,N},Array{T,N},Array{T,N}}(zeros(T,I...),zeros(T,I...),zeros(T,I...))
    
    function VecArray(;x::Union{Zero,AbstractArray}=𝟎,y::Union{Zero,AbstractArray}=𝟎, z::Union{Zero,AbstractArray}=𝟎)
        if (x,y,z) === (𝟎,𝟎,𝟎)
            throw(DomainError((x,y,z),"At least one entry must be a valid array"))
        else
            if x !== 𝟎
                s = size(x)
                xv = x
                yv = y === 𝟎 ? ZeroArray(s) : y
                zv = z === 𝟎 ? ZeroArray(s) : z
                size(yv) === s || throw(DomainError((x,y),"Arrays must have the same size"))
                size(zv) === s || throw(DomainError((x,z),"Arrays must have the same size"))

                Tx = typeof(xv)
                Ty = typeof(yv)
                Tz = typeof(zv)
                N = ndims(xv)

            elseif y !== 𝟎
                s = size(y)
                xv = ZeroArray(s)
                yv = y
                zv = z === 𝟎 ? ZeroArray(s) : z
                size(zv) === s || throw(DomainError((y,z),"Arrays must have the same size"))

                Tx = typeof(xv)
                Ty = typeof(yv)
                Tz = typeof(zv)
                N = ndims(yv)
            else # z must be !== 𝟎
                s = size(z)
                xv = ZeroArray(s)
                yv = ZeroArray(s)
                zv = z 

                Tx = typeof(xv)
                Ty = typeof(yv)
                Tz = typeof(zv)
                N = ndims(zv)
            end
            return new{_vec_type(eltype(xv),eltype(yv),eltype(zv)),N,Tx,Ty,Tz}(xv,yv,zv)
        end
    end

end

const Vec3DArray{T,N} = VecArray{Vec3D{T},N,Array{T,N},Array{T,N},Array{T,N}}
const Vec2DxyArray{T,N} = VecArray{Vec2Dxy{T},N,Array{T,N},Array{T,N},ZeroArray{N}}
const Vec2DxzArray{T,N} = VecArray{Vec2Dxz{T},N,Array{T,N},ZeroArray{N},Array{T,N}}
const Vec2DyzArray{T,N} = VecArray{Vec2Dyz{T},N,ZeroArray{N},Array{T,N},Array{T,N}}
const Vec1DxArray{T,N} = VecArray{Vec1Dx{T},N,Array{T,N},ZeroArray{N},ZeroArray{N}}
const Vec1DyArray{T,N} = VecArray{Vec1Dy{T},N,ZeroArray{N},Array{T,N},ZeroArray{N}}
const Vec1DzArray{T,N} = VecArray{Vec1Dz{T},N,ZeroArray{N},ZeroArray{N},Array{T,N}}

@inline Base.size(A::VecArray) = size(A.x)
@inline Base.length(A::VecArray) = length(A.x)

@inline function Base.getindex(A::VecArray,i::Int)
    @boundscheck checkbounds(A,i)
    @inbounds r = Vec(x=A.x[i], y=A.y[i], z=A.z[i])
    return r
end

@inline function Base.getindex(A::VecArray{T,N},I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A,I...)
    @inbounds r = Vec(x=A.x[I...], y=A.y[I...], z=A.z[I...])
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

Base.similar(A::VecArray,T::Type{Vec{Tt,N,Tx,Ty,Tz}},dims::Tuple{Int,Vararg{Int,N2}}) where {Tt,N,Tx,Ty,Tz,N2} = VecArray(x=similar(A.x,Tx,dims), y=similar(A.y,Ty,dims), z=similar(A.z,Tz,dims))
