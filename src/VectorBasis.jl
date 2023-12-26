module VectorBasis

using Zeros
using MuladdMacro

export Vec, Vec0D, Vec1D, Vec2D, Vec3D, xx, yy, zz, xvec, yvec, zvec
export 𝐢, 𝐣, 𝐤

abstract type AbstractVec{T} <: AbstractVector{T} end

Base.eltype(::AbstractVec{T}) where T = Union{T,_zero_type(T)}
@inline _zero_type(::Type{<:Number}) = Zeros.Zero
@inline _zero(::Type{<:Number}) = Zeros.Zero()

Base.length(::Type{<:AbstractVec{<:Number}}) = 3
Base.length(u::AbstractVec{<:Number}) = length(typeof(u))

Base.size(::Type{<:AbstractVec{<:Number}}) = (3,)
Base.size(u::AbstractVec{<:Number}) = size(typeof(u))

Base.length(::Type{<:AbstractVec{AbstractVec{<:Number}}}) = 9
Base.length(u::AbstractVec{AbstractVec{<:Number}}) = length(typeof(u))

Base.size(::Type{<:AbstractVec{AbstractVec{<:Number}}}) = (3,3)
Base.size(u::AbstractVec{AbstractVec{<:Number}}) = size(typeof(u))


@inline xvec(u::AbstractVec{T}) where T = Vec0D{T}()
@inline yvec(u::AbstractVec{T}) where T = Vec0D{T}()
@inline zvec(u::AbstractVec{T}) where T = Vec0D{T}()
@inline xx(u::AbstractVec) = xx(xvec(u))
@inline yy(u::AbstractVec) = yy(yvec(u))
@inline zz(u::AbstractVec) = zz(zvec(u))

function Base.getproperty(u::AbstractVec,s::Symbol)
    if s === :x || s === :i
        return xx(u)
    elseif s === :y || s === :j
        return yy(u)
    elseif s === :z || s === :k
        return zz(u)
    else
        return getfield(u,s)
    end
end

struct Vec0D{T} <: AbstractVec{T} end

@inline function Base.getindex(u::Vec0D{T},i::Integer) where T
    @boundscheck checkbounds(u,i)
    return _zero(T)
end

Base.eltype(::Vec0D{T}) where T = _zero_type(T)

const 𝟎⃗ = Vec0D{Zero}()

@inline xx(::Vec0D{T}) where T = _zero(T)
@inline yy(::Vec0D{T}) where T = _zero(T)
@inline zz(::Vec0D{T}) where T = _zero(T)


@inline test_N(::Val{1}) = true
@inline test_N(::Val{2}) = true
@inline test_N(::Val{3}) = true
@inline test_N(::Val) = false

struct Vec1D{N,T} <: AbstractVec{T}
    v::T

    @inline function Vec1D{N,T}(v::T) where {N,T}
        if test_N(Val{N}())
            if T === Zero
                return 𝟎⃗
            else
                return new{N,T}(v)
            end
        else
            throw(DomainError(N,"Dimension invalid"))
        end
    end

end

@inline Vec1D{N}(v::T) where{N,T} = Vec1D{N,T}(v)
@inline Vec1D(v) = Vec1D{1}(v)


@inline function Base.getindex(u::Vec1D{N,T},i::Integer) where {N,T}
    @boundscheck checkbounds(u,i)
    return i === N ? u.v : _zero(T)
end

@inline xvec(u::Vec1D{1}) = u
@inline yvec(u::Vec1D{2}) = u
@inline zvec(u::Vec1D{3}) = u
@inline xx(u::Vec1D{1}) = getfield(u,1)
@inline yy(u::Vec1D{2}) = getfield(u,1)
@inline zz(u::Vec1D{3}) = getfield(u,1)

@inline _zero_type(::Type{<:Vec1D}) = Vec0D{Zero}
@inline _zero(::Type{<:Vec1D}) = 𝟎⃗

test_swap(::Val,::Val) = false
test_swap(::Val{2},::Val{1}) = true
test_swap(::Val{3},::Val{1}) = true
test_swap(::Val{3},::Val{2}) = true

struct Vec2D{N1,N2,T} <: AbstractVec{T}
    v1::Vec1D{N1,T}
    v2::Vec1D{N2,T}

    @inline function Vec2D{N1,N2,T}(v1::Vec1D{N1,T},v2::Vec1D{N2,T}) where {N1,N2,T}
        if test_N(Val{N1}()) && test_N(Val{N2}())
            if Val{N1}() !== Val{N2}()
                if test_swap(Val{N1}(), Val{N2}())
                    return new{N2,N1,T}(v2,v1)
                else
                    return new{N1,N2,T}(v1,v2)
                end
            else
                throw(DomainError((N1,N2),"Dimensions invalid"))
            end
        else
            throw(DomainError((N1,N2),"Dimensions invalid"))
        end
    end

    @inline Vec2D{N1,N2,Zero}(v1,v2) where {N1,N2} = 𝟎⃗
end

@inline Vec2D{N1,N2}(v1::Vec1D{N1,T},v2::Vec1D{N2,T}) where {N1,N2,T} = Vec2D{N1,N2,T}(v1,v2)
@inline function Vec2D(v1::Vec1D{N1,T1},v2::Vec1D{N2,T2}) where {N1,N2,T1,T2} 
    T = promote_type(T1,T2)
    return Vec2D{N1,N2}(Vec1D{N1}(T(v1.v)), Vec1D{N2}(T(v2.v)))
end
@inline Vec2D(v1::Vec1D{N},v2::Vec0D) where {N} = Vec1D{N}(v1.v)
@inline Vec2D(v2::Vec0D,v1::Vec1D{N}) where {N} = Vec1D{N}(v1.v)

@inline function Base.getindex(u::Vec2D{N1,N2,T},i::Integer) where {N1,N2,T}
    @boundscheck checkbounds(u,i)
    return i === N1 ? u.v1.v : i === N2 ? u.v2.v : _zero(T)
end

@inline xvec(u::Vec2D{1}) = getfield(u,1)
@inline yvec(u::Vec2D{1,2}) = getfield(u,2)
@inline yvec(u::Vec2D{2,3}) = getfield(u,1)
@inline zvec(u::Vec2D{N,3}) where N = getfield(u,2)

@inline _zero_type(::Type{<:Vec2D}) = Vec0D{Zero}
@inline _zero(::Type{<:Vec2D}) = 𝟎⃗

struct Vec3D{T} <: AbstractVec{T}
    v1::Vec1D{1,T}
    v2::Vec1D{2,T}
    v3::Vec1D{3,T}
end

@inline function Vec3D(v1::Vec1D{1,T1}, v2::Vec1D{2,T2}, v3::Vec1D{3,T3}) where {T1,T2,T3} 
    T = promote_type(T1,T2,T3)
    return Vec3D(Vec1D{1}(T(v1.v)), Vec1D{2}(T(v2.v)), Vec1D{3}(T(v3.v)))
end

@inline function Base.getindex(u::Vec3D,i::Integer)
    @boundscheck checkbounds(u,i)
    return getfield(getfield(u,i),1)
end

@inline xvec(u::Vec3D) = getfield(u,1)
@inline yvec(u::Vec3D) = getfield(u,2)
@inline zvec(u::Vec3D) = getfield(u,3)

@inline _zero_type(::Type{<:Vec3D}) = Vec0D{Zero}
@inline _zero(::Type{<:Vec3D}) = 𝟎⃗

const 𝐢 = Vec1D{1}(One())
const 𝐣 = Vec1D{2}(One())
const 𝐤 = Vec1D{3}(One())

struct Vec{N1,N2} end
@inline Vec() = 𝟎⃗
@inline Vec(x) = Vec1D(x)
@inline Vec{N}(x) where N = Vec1D{N}(x)
@inline Vec(x::Zero) = Vec0D{Zero}()
@inline Vec{N}(x::Zero) where N = Vec0D{Zero}()
@inline Vec(x,y) = Vec2D(Vec1D{1}(x),Vec1D{2}(y))
@inline Vec(x,y::Zero) = Vec1D{1}(x)
@inline Vec(x::Zero,y) = Vec1D{2}(y)
@inline Vec{N1,N2}(x,y) where {N1,N2} = Vec2D(Vec{N1}(x),Vec{N2}(y))
@inline Vec{N1,N2}(x,y::Zero) where {N1,N2} = Vec{N1}(x)
@inline Vec{N1,N2}(x::Zero,y) where {N1,N2} = Vec{N2}(y)
@inline Vec(x,y,z) = Vec3D(Vec1D{1}(x),Vec1D{2}(y),Vec1D{3}(z))
@inline Vec(x::Zero,y,z) = Vec2D(Vec1D{2}(y),Vec1D{3}(z))
@inline Vec(x::Zero,y::Zero,z) = Vec1D{3}(z)
@inline Vec(x::Zero,y::Zero,z::Zero) = Vec0D{Zero}()
@inline Vec(x,y::Zero,z) = Vec2D(Vec1D{1}(x),Vec1D{3}(z))
@inline Vec(x,y::Zero,z::Zero) = Vec1D{1}(x)
@inline Vec(x,y,z::Zero) = Vec2D(Vec1D{1}(x),Vec1D{2}(y))
@inline Vec(x::Zero,y,z::Zero) = Vec1D{2}(y)


include("vec_arithmetic.jl")

end
