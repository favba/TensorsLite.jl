struct VecArray{T, N, Tx, Ty, Tz} <: AbstractArray{T, N}
    x::Tx
    y::Ty
    z::Tz

    VecArray{T}(I::Vararg{Integer, N}) where {T, N} = new{Vec{T, 1, T, T, T}, N, Array{T, N}, Array{T, N}, Array{T, N}}(Array{T}(undef, I...), Array{T}(undef, I...), Array{T}(undef, I...))

    function VecArray(x::AbstractArray, y::AbstractArray, z::AbstractArray)

        s = size(x)
        size(y) === s || throw(DimensionMismatch("Arrays must have the same size"))
        size(z) === s || throw(DimensionMismatch("Arrays must have the same size"))

        Tx = typeof(x)
        Ty = typeof(y)
        Tz = typeof(z)
        N = ndims(x)
        eTx = eltype(x)
        eTy = eltype(y)
        eTz = eltype(z)

        return new{Vec{Union{eTx, eTy, eTz}, 1, eTx, eTy, eTz}, N, Tx, Ty, Tz}(x, y, z)
    end

    function VecArray(x::AbstractArray{Tx, N}, y::AbstractArray{Ty, N}, z::AbstractArray{Tz, N}) where {NV, Tx<:AbstractVec{<:Any, NV}, Ty<:AbstractVec{<:Any, NV}, Tz<:AbstractVec{<:Any, NV}, N}

        s = size(x)
        size(y) === s || throw(DimensionMismatch("Arrays must have the same size"))
        size(z) === s || throw(DimensionMismatch("Arrays must have the same size"))

        eTx = eltype(x)
        eeTx = eltype(eTx)
        eTy = eltype(y)
        eeTy = eltype(eTy)
        eTz = eltype(z)
        eeTz = eltype(eTz)

        return new{Vec{Union{eeTx, eeTy, eeTz}, NV+1, eTx, eTy, eTz}, N, typeof(x), typeof(y), typeof(z)}(x, y, z)
    end

end

@inline ZeroVecArray(I::Vararg{Integer, NI}) where {NI} = VecArray(Array{Zero}(undef, I...), Array{Zero}(undef, I...), Array{Zero}(undef, I...) )
@inline ZeroVecArray(::Val{1}, I::Vararg{Integer, NI}) where {NI} = ZeroVecArray(I...)
@inline ZeroVecArray(::Val{N}, I::Vararg{Integer, NI}) where {N, NI} = VecArray(ZeroVecArray(Val{N-1}(), I...), ZeroVecArray(Val{N-1}(), I...), ZeroVecArray(Val{N-1}(), I...))


const Vec3DArray{T, N} = VecArray{Vec3D{T}, N, Array{T, N}, Array{T, N}, Array{T, N}}

Vec3DArray(a::AbstractArray{T,N}, b::AbstractArray{T,N}, c::AbstractArray{T,N}) where {T,N} = VecArray(a, b, c)


const Vec2DxyArray{T, N} = VecArray{Vec2Dxy{T}, N, Array{T, N}, Array{T, N}, Array{Zero, N}}

Vec2DxyArray(a::AbstractArray{T,N}, b::AbstractArray{T,N}) where {T,N} = VecArray(a, b, similar(a, Zero))

Vec2DxyArray(a::AbstractArray{<:AbstractVec{TV,NV},N}, b::AbstractArray{<:AbstractVec{TV,NV},N}) where {TV, NV, N} = VecArray(a, b, ZeroVecArray(Val{NV}(), size(a)...))


const Vec2DxzArray{T, N} = VecArray{Vec2Dxz{T}, N, Array{T, N}, Array{Zero, N}, Array{T, N}}

Vec2DxzArray(a::AbstractArray{T,N}, b::AbstractArray{T,N}) where {T,N} = VecArray(a, similar(a, Zero), b)

Vec2DxzArray(a::AbstractArray{<:AbstractVec{TV,NV},N}, b::AbstractArray{<:AbstractVec{TV,NV},N}) where {TV, NV, N} = VecArray(a, ZeroVecArray(Val{NV}(), size(a)...), b)


const Vec2DyzArray{T, N} = VecArray{Vec2Dyz{T}, N, Array{Zero, N}, Array{T, N}, Array{T, N}}

Vec2DyzArray(a::AbstractArray{T,N}, b::AbstractArray{T,N}) where {T,N} = VecArray(similar(a, Zero), a, b)

Vec2DyzArray(a::AbstractArray{<:AbstractVec{TV,NV},N}, b::AbstractArray{<:AbstractVec{TV,NV},N}) where {TV, NV, N} = VecArray(ZeroVecArray(Val{NV}(), size(a)...), a, b)


const Vec2DArray{T, N} = Union{Vec2DxyArray{T, N}, Vec2DxzArray{T, N}, Vec2DyzArray{T, N}}


const Vec1DxArray{T, N} = VecArray{Vec1Dx{T}, N, Array{T, N}, Array{Zero, N}, Array{Zero, N}}

Vec1DxArray(a::AbstractArray{T,N}) where {T,N} = VecArray(a, similar(a, Zero), similar(a, Zero))

Vec1DxArray(a::AbstractArray{<:AbstractVec{TV,NV},N}) where {TV, NV, N} = VecArray(a, ZeroVecArray(Val{NV}(), size(a)...), ZeroVecArray(Val{NV}(), size(a)...))


const Vec1DyArray{T, N} = VecArray{Vec1Dy{T}, N, Array{Zero, N}, Array{T, N}, Array{Zero, N}}

Vec1DyArray(a::AbstractArray{T,N}) where {T,N} = VecArray(similar(a, Zero), a, similar(a, Zero))

Vec1DyArray(a::AbstractArray{<:AbstractVec{TV,NV},N}) where {TV, NV, N} = VecArray(ZeroVecArray(Val{NV}(), size(a)...), a, ZeroVecArray(Val{NV}(), size(a)...))


const Vec1DzArray{T, N} = VecArray{Vec1Dz{T}, N, Array{Zero, N}, Array{Zero, N}, Array{T, N}}

Vec1DzArray(a::AbstractArray{T,N}) where {T,N} = VecArray(similar(a, Zero), similar(a, Zero), a)

Vec1DzArray(a::AbstractArray{<:AbstractVec{TV,NV},N}) where {TV, NV, N} = VecArray(ZeroVecArray(Val{NV}(), size(a)...), ZeroVecArray(Val{NV}(), size(a)...), a)


const Vec1DArray{T, N} = Union{Vec1DxArray{T, N}, Vec1DyArray{T, N}, Vec1DzArray{T, N}}
const Vec0DArray{N} = VecArray{Vec0D, N, Array{Zero, N}, Array{Zero, N}, Array{Zero, N}}
const VecNDArray{T, N} = Union{Vec3DArray{T, N}, Vec2DArray{T, N}, Vec1DArray{T, N}}

const VecMaybe2DxyArray{T, Tz, N} = VecArray{VecMaybe2Dxy{T, Tz}, N, Array{T, N}, Array{T, N}, Array{Tz, N}}

@inline function TenArray(xx::AbstractArray, xy::AbstractArray, xz::AbstractArray,
                          yx::AbstractArray, yy::AbstractArray, yz::AbstractArray,
                          zx::AbstractArray, zy::AbstractArray, zz::AbstractArray)
    x = VecArray(xx, yx, zx)
    y = VecArray(xy, yy, zy)
    z = VecArray(xz, yz, zz)
    return VecArray(x, y, z)
end

_if_zero_to_Array(s::NTuple{N, Int}, ::Zero) where {N} = Array{Zero}(undef, s)
_if_zero_to_Array(s::NTuple{N, Int}, x::AbstractArray) where {N} = x

function TenArray(;
        xx = 𝟎, xy = 𝟎, xz = 𝟎,
        yx = 𝟎, yy = 𝟎, yz = 𝟎,
        zx = 𝟎, zy = 𝟎, zz = 𝟎,
    )

    vals = (
        xx, yx, zx,
        xy, yy, zy,
        xz, yz, zz,
    )

    vals === (
        𝟎, 𝟎, 𝟎,
        𝟎, 𝟎, 𝟎,
        𝟎, 𝟎, 𝟎,
    ) && throw(DomainError(vals, "At least one entry must be a valid Array"))

    non_zero_vals = _filter_zeros(vals...)
    s = size(non_zero_vals[1])

    all(x -> (size(x) === s), non_zero_vals) || throw(DimensionMismatch())

    sizes = (s, s, s, s, s, s, s, s, s)
    final_vals = map(_if_zero_to_Array, sizes, vals)

    xv = VecArray(final_vals[1], final_vals[2], final_vals[3])
    yv = VecArray(final_vals[4], final_vals[5], final_vals[6])
    zv = VecArray(final_vals[7], final_vals[8], final_vals[9])

    return VecArray(xv, yv, zv)
end

ZeroArray(s) = Array{Zero}(undef, s)

const Ten3DArray{T, N} = VecArray{Ten3D{T}, N, Vec3DArray{T, N}, Vec3DArray{T, N}, Vec3DArray{T, N}}

const Ten2DxyArray{T, N} = VecArray{Ten2Dxy{T}, N, Vec2DxyArray{T, N}, Vec2DxyArray{T, N}, Vec0DArray{N}}
Ten2DxyArray(xx::AbstractArray, xy::AbstractArray, yx::AbstractArray, yy::AbstractArray) = 
    TenArray(xx,     xy,     ZeroArray(size(xx)),
             yx,     yy,     ZeroArray(size(xx)),
             ZeroArray(size(xx)), ZeroArray(size(xx)), ZeroArray(size(xx)))

const Ten2DxzArray{T, N} = VecArray{Ten2Dxz{T}, N, Vec2DxzArray{T, N}, Vec0DArray{N}, Vec2DxzArray{T, N}}

Ten2DxzArray(xx, xz, zx, zz) = 
    TenArray(xx,     ZeroArray(size(xx)), xz,
             ZeroArray(size(xx)), ZeroArray(size(xx)), ZeroArray(size(xx)),
             zx,     ZeroArray(size(xx)), zz)

const Ten2DyzArray{T, N} = VecArray{Ten2Dyz{T}, N, Vec0DArray{N}, Vec2DyzArray{T, N}, Vec2DyzArray{T, N}}
Ten2DyzArray(yy, yz, zy, zz) = 
    TenArray(ZeroArray(size(yy)), ZeroArray(size(yy)), ZeroArray(size(yy)),
             ZeroArray(size(yy)), yy,     yz,
             ZeroArray(size(yy)), zy,     zz)

const Ten2DArray{T, N} = Union{Ten2DxyArray{T, N}, Ten2DxzArray{T, N}, Ten2DyzArray{T, N}}

const Ten1DxArray{T, N} = VecArray{Ten1Dx{T}, N, Vec1DxArray{T, N}, Vec0DArray{N}, Vec0DArray{N}}
Ten1DxArray(xx) = 
    TenArray(xx, ZeroArray(size(xx)), ZeroArray(size(xx)),
             ZeroArray(size(xx)), ZeroArray(size(xx)), ZeroArray(size(xx)),
             ZeroArray(size(xx)), ZeroArray(size(xx)), ZeroArray(size(xx)))

const Ten1DyArray{T, N} = VecArray{Ten1Dy{T}, N, Vec0DArray{N}, Vec1DyArray{T, N}, Vec0DArray{N}}
Ten1DyArray(yy) = 
    TenArray(ZeroArray(size(yy)), ZeroArray(size(yy)), ZeroArray(size(yy)),
             ZeroArray(size(yy)), yy, ZeroArray(size(yy)),
             ZeroArray(size(yy)), ZeroArray(size(yy)), ZeroArray(size(yy)))


const Ten1DzArray{T, N} = VecArray{Ten1Dz{T}, N, Vec0DArray{N}, Vec0DArray{N}, Vec1DzArray{T, N}}
Ten1DzArray(zz) = 
    TenArray(ZeroArray(size(zz)), ZeroArray(size(zz)), ZeroArray(size(zz)),
             ZeroArray(size(zz)), ZeroArray(size(zz)), ZeroArray(size(zz)),
             ZeroArray(size(zz)), ZeroArray(size(zz)), zz)

const Ten1DArray{T, N} = Union{Ten1DxArray{T, N}, Ten1DyArray{T, N}, Ten1DzArray{T, N}}
const TenNDArray{T, N} = Union{Ten3DArray{T, N}, Ten2DArray{T, N}, Ten1DArray{T, N}}

const TenMaybe2DxyArray{T, Tz, N} = VecArray{TenMaybe2Dxy{T, Tz}, N, VecMaybe2DxyArray{T, Tz, N}, VecMaybe2DxyArray{T, Tz, N}, Vec3DArray{Tz, N}}

@inline Base.size(A::VecArray) = size(A.x)
@inline Base.length(A::VecArray) = length(A.x)

Base.dataids(A::VecArray) = (Base.dataids(A.x)..., Base.dataids(A.y)..., Base.dataids(A.z)...)

@inline function Base.getindex(A::VecArray, i::Int)
    @boundscheck checkbounds(A, i)
    @inbounds @inline r = Vec(A.x[i], A.y[i], A.z[i])
    return r
end

@inline function Base.getindex(A::VecArray{T, N}, I::Vararg{Int, N}) where {T, N}
    @boundscheck checkbounds(A, I...)
    @inbounds @inline r = Vec(A.x[I...], A.y[I...], A.z[I...])
    return r
end

@inline function Base.setindex!(A::VecArray, u::AbstractVec, i::Int)
    @boundscheck checkbounds(A, i)

    @inbounds @inline begin
        A.x[i] = u.x
        A.y[i] = u.y
        A.z[i] = u.z
    end

    return A
end

@inline function Base.setindex!(A::VecArray{T, N}, u::AbstractVec, I::Vararg{Int, N}) where {T, N}
    @boundscheck checkbounds(A, I...)

    @inbounds @inline begin
        A.x[I...] = u.x
        A.y[I...] = u.y
        A.z[I...] = u.z
    end

    return A
end

Base.similar(A::VecArray, ::Type{Vec{Tt, N, Tx, Ty, Tz}}, dims::Tuple{Int, Vararg{Int, N2}}) where {Tt, N, Tx, Ty, Tz, N2} = VecArray(similar(A.x, Tx, dims), similar(A.y, Ty, dims), similar(A.z, Tz, dims))

Base.resize!(A::VecArray{T,1}, i::Integer) where T = begin resize!(A.x, i); resize!(A.y, i); resize!(A.z, i); A end

#Definitons so broadcast return a VecArray =======================================

function Base.similar(bc::Broadcast.Broadcasted, ::Type{Vec{T, 1, Tx, Ty, Tz}}) where {T, Tx, Ty, Tz}
    s = length.(axes(bc))
    x = Array{Tx}(undef, s...)
    y = Array{Ty}(undef, s...)
    z = Array{Tz}(undef, s...)
    return VecArray(x, y, z)
end

function Base.similar(bc::Broadcast.Broadcasted, ::Type{Vec{T, 2, Tx, Ty, Tz}}) where {T, Tx, Ty, Tz}
    xv = similar(bc, Tx)
    yv = similar(bc, Ty)
    zv = similar(bc, Tz)
    return VecArray(xv, yv, zv)
end

#Definitons so broadcast return a VecArray =======================================

@inline function Base.getproperty(T::TenNDArray, s::Symbol)
    if s === :xx
        return getfield(getfield(T, :x), :x)
    elseif s === :xy
        return getfield(getfield(T, :y), :x)
    elseif s === :xz
        return getfield(getfield(T, :z), :x)
    elseif s === :yx
        return getfield(getfield(T, :x), :y)
    elseif s === :yy
        return getfield(getfield(T, :y), :y)
    elseif s === :yz
        return getfield(getfield(T, :z), :y)
    elseif s === :zx
        return getfield(getfield(T, :x), :z)
    elseif s === :zy
        return getfield(getfield(T, :y), :z)
    elseif s === :zz
        return getfield(getfield(T, :z), :z)
    else
        return getfield(T, s)
    end
end
