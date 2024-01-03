struct ZeroVec{N} <: AbstractArray{Zero,N}
    size::NTuple{N,Int}
end
Base.size(z::ZeroVec) = z.size

@inline Base.getindex(z::ZeroVec{N},I::Vararg{Int,N}) where N = 𝟎
@inline Base.setindex!(z::ZeroVec{N},x,I::Vararg{Int,N}) where N = begin convert(Zero,x); return nothing end

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
                yv = y === 𝟎 ? ZeroVec(s) : y
                zv = z === 𝟎 ? ZeroVec(s) : z
                size(yv) === s || throw(DomainError((x,y),"Arrays must have the same size"))
                size(zv) === s || throw(DomainError((x,z),"Arrays must have the same size"))

                Tx = typeof(xv)
                Ty = typeof(yv)
                Tz = typeof(zv)
                N = ndims(xv)

            elseif y !== 𝟎
                s = size(y)
                xv = ZeroVec(s)
                yv = y
                zv = z === 𝟎 ? ZeroVec(s) : z
                size(zv) === s || throw(DomainError((y,z),"Arrays must have the same size"))

                Tx = typeof(xv)
                Ty = typeof(yv)
                Tz = typeof(zv)
                N = ndims(yv)
            else # z must be !== 𝟎
                s = size(z)
                xv = ZeroVec(s)
                yv = ZeroVec(s)
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

@inline Base.size(A::VecArray) = size(A.x)
@inline Base.length(A::VecArray) = length(A.x)

@inline function Base.getindex(A::VecArray{T,N},I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A,I...)
    @inbounds r = Vec(x=A.x[I...], y=A.y[I...], z=A.z[I...])
    return r
end

@inline function Base.setindex!(A::VecArray{T,N}, u::AbstractVec, I::Vararg{Int,N}) where {T,N}
    @boundscheck checkbounds(A,I...)

    @inbounds begin 
        A.x[I...] = u.x
        A.y[I...] = u.y
        A.z[I...] = u.z
    end

    return nothing
end