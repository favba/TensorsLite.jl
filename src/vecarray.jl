abstract type AbstractTensorArray{T, N} <: AbstractArray{T, N} end

const VecArray{T,N,Tx,Ty,Tz} = AbstractTensorArray{Tensor{1,T,Tx,Ty,Tz},N}

const TenArray{T,N,Tx,Ty,Tz} = AbstractTensorArray{Tensor{2,T,Tx,Ty,Tz},N}

struct TensorArray{T, N, Tx, Ty, Tz} <: AbstractTensorArray{T, N}
    x::Tx
    y::Ty
    z::Tz

    TensorArray{T,1}(I::Vararg{Integer, N}) where {T, N} = new{Tensor{1, T, T, T, T}, N, Array{T, N}, Array{T, N}, Array{T, N}}(Array{T}(undef, I...), Array{T}(undef, I...), Array{T}(undef, I...))

    function TensorArray{T,NT}(I::Vararg{Integer, N}) where {T,NT,N} 
        x = TensorArray{T,NT-1}(I...)
        y = TensorArray{T,NT-1}(I...)
        z = TensorArray{T,NT-1}(I...)
        Tf = typeof(x)
        new{tensor_type_3D(Val{NT}(),T),N,Tf,Tf,Tf}(x,y,z)
    end

    function TensorArray(x::AbstractArray, y::AbstractArray, z::AbstractArray)

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
        Tf = promote_type_ignoring_Zero_and_One(eTx, eTy, eTz)
        fTx = _my_promote_type(Tf, eTx)
        fTy = _my_promote_type(Tf, eTy)
        fTz = _my_promote_type(Tf, eTz)
        Tff = Union{fTx,fTy,fTz}

        return new{Tensor{1, Tff, fTx, fTy, fTz}, N, Tx, Ty, Tz}(x, y, z)
    end

    function TensorArray(x::AbstractArray{Tx, N}, y::AbstractArray{Ty, N}, z::AbstractArray{Tz, N}) where {NV, eeTx, eeTy, eeTz, Tx<:AbstractTensor{NV, eeTx}, Ty<:AbstractTensor{NV, eeTy}, Tz<:AbstractTensor{NV, eeTz}, N}

        s = size(x)
        size(y) === s || throw(DimensionMismatch("Arrays must have the same size"))
        size(z) === s || throw(DimensionMismatch("Arrays must have the same size"))

        # Figure out how to not rely on promote_op
        TT = Base.promote_op(Tensor,Tx,Ty,Tz)

        return new{TT, N, typeof(x), typeof(y), typeof(z)}(x, y, z)
    end

end

### Aliases ####

const Vec3DArray{T,N} = TensorArray{Vec3D{T},N,Array{T,N},Array{T,N},Array{T,N}}
const Vec2DxyArray{T,N} = TensorArray{Vec2Dxy{T},N,Array{T,N},Array{T,N},Array{Zero,N}}
const Vec2DxzArray{T,N} = TensorArray{Vec2Dxz{T},N,Array{T,N},Array{Zero,N},Array{T,N}}
const Vec2DyzArray{T,N} = TensorArray{Vec2Dyz{T},N,Array{Zero,N},Array{T,N},Array{T,N}}
const Vec2DArray{T,N} = Union{Vec2DxyArray{T, N}, Vec2DxzArray{T, N},Vec2DyzArray{T, N}}
const Vec1DxArray{T,N} = TensorArray{Vec1Dx{T},N,Array{T,N},Array{Zero,N},Array{Zero,N}}
const Vec1DyArray{T,N} = TensorArray{Vec1Dy{T},N,Array{Zero,N},Array{T,N},Array{Zero,N}}
const Vec1DzArray{T,N} = TensorArray{Vec1Dz{T},N,Array{Zero,N},Array{Zero,N},Array{T,N}}
const Vec1DArray{T,N} = Union{Vec1DxArray{T, N}, Vec1DyArray{T, N}, Vec1DzArray{T, N}}
const VecNDArray{T,N} = Union{Vec3DArray{T, N}, Vec2DArray{T, N}, Vec1DArray{T, N}}
const Vec0DArray{N} = Vec3DArray{Zero,N}

const Ten3DArray{T,N} = TensorArray{Ten3D{T},N,Vec3DArray{T,N},Vec3DArray{T,N},Vec3DArray{T,N}}
const Ten2DxyArray{T,N} = TensorArray{Ten2Dxy{T},N,Vec2DxyArray{T,N},Vec2DxyArray{T,N},Vec0DArray{N}}
const Ten2DxzArray{T,N} = TensorArray{Ten2Dxz{T},N,Vec2DxzArray{T,N},Vec0DArray{N},Vec2DxzArray{T,N}}
const Ten2DyzArray{T,N} = TensorArray{Ten2Dyz{T},N,Vec0DArray{N},Vec2DyzArray{T,N},Vec2DyzArray{T,N}}
const Ten2DArray{T, N} = Union{Ten2DxyArray{T, N}, Ten2DxzArray{T, N}, Ten2DyzArray{T, N}}
const Ten1DxArray{T,N} = TensorArray{Ten1Dx{T},N,Vec1DxArray{T,N},Vec0DArray{N},Vec0DArray{N}}
const Ten1DyArray{T,N} = TensorArray{Ten1Dy{T},N,Vec0DArray{N},Vec1DyArray{T,N},Vec0DArray{N}}
const Ten1DzArray{T,N} = TensorArray{Ten1Dz{T},N,Vec0DArray{N},Vec0DArray{N},Vec1DyArray{T,N}}
const Ten1DArray{T, N} = Union{Ten1DxArray{T, N}, Ten1DyArray{T, N}, Ten1DzArray{T, N}}
const TenNDArray{T, N} = Union{Ten3DArray{T, N}, Ten2DArray{T, N}, Ten1DArray{T, N}}

const VecMaybe2DxyArray{T, Tz, N} = TensorArray{VecMaybe2Dxy{T, Tz}, N, Array{T, N}, Array{T, N}, Array{Tz, N}}
const TenMaybe2DxyArray{T, Tz, N} = TensorArray{TenMaybe2Dxy{T, Tz}, N, VecMaybe2DxyArray{T, Tz, N}, VecMaybe2DxyArray{T, Tz, N}, Vec3DArray{Tz, N}}

############### Constructors ###################

ZeroArray(s) = Array{Zero}(undef, s)

@inline ZeroTensorArray(I::Vararg{Integer, NI}) where {NI} = TensorArray(ZeroArray(I), ZeroArray(I), ZeroArray(I))
@inline ZeroTensorArray(::Val{0}, I::Vararg{Integer, NI}) where {NI} = ZeroArray(I)
@inline ZeroTensorArray(::Val{1}, I::Vararg{Integer, NI}) where {NI} = TensorArray(ZeroArray(I), ZeroArray(I), ZeroArray(I))
@inline ZeroTensorArray(::Val{N}, I::Vararg{Integer, NI}) where {N, NI} = TensorArray(ZeroTensorArray(Val{N-1}(), I...), ZeroTensorArray(Val{N-1}(), I...), ZeroTensorArray(Val{N-1}(), I...))

_check_TensorArray_args(v::Vararg) = any(map(x->isa(x,AbstractArray{<:AbstractTensor}), v)) ? throw(DomainError("Tensor Array fields must be Arrays of  tensors with the same order")) : nothing
_check_TensorArray_args(::Vararg{T}) where {TE, N, T<:AbstractArray{<:AbstractTensor{N,TE}}} = nothing
check_TensorArray_args(x,y,z) = _check_TensorArray_args(_filter_zeros(x,y,z)...)

_get_ndims(::Union{Zero,<:AbstractArray{<:AbstractTensor{N}}},
           ::Union{Zero,<:AbstractArray{<:AbstractTensor{N}}},
           ::Union{Zero,<:AbstractArray{<:AbstractTensor{N}}}) where {N} = Val{N}()

_get_size(v::Vararg) = size(v[1])
get_size(x,y,z) = _get_size(_filter_zeros(x,y,z)...)

function if_zero_to_ZeroTensorArray(v::Val{NV}, s::NTuple{N,Int}, x, y, z) where {NV, N}
    xf = isa(x,Zero) ? ZeroTensorArray(v, s...) : x
    yf = isa(y,Zero) ? ZeroTensorArray(v, s...) : y
    zf = isa(z,Zero) ? ZeroTensorArray(v, s...) : z
    return (xf, yf, zf)
end

@inline function TensorArray(;x=𝟎,y=𝟎,z=𝟎)
    if (x === 𝟎) && (y === 𝟎) && (z === 𝟎)
        return throw(DomainError((x,y,z), "At least one entry must be a valid Array"))
    else
        check_TensorArray_args(x,y,z)
        NV = _get_ndims(x,y,z)
        s = get_size(x,y,z)
        xf, yf, zf = if_zero_to_ZeroTensorArray(NV,s, x,y,z)
        return TensorArray(xf,yf,zf)
    end
end

tensor_ndims(::Type{T}) where {T} = Val{0}()
tensor_ndims(::Type{TV}) where {T,N,TV<:Tensor{N,T}} = Val{N}()

Tensor2DxyArray(a::AbstractArray{T,N},b::AbstractArray{T,N}) where {T,N} = TensorArray(a, b, ZeroTensorArray(tensor_ndims(T), size(a)...))
Tensor2DxyArray(::Type{T}, ::Val{1}, I::Vararg{Integer,N}) where {T, N} = Tensor2DxyArray(Array{T}(undef, I...), Array{T}(undef, I...))
Tensor2DxyArray(::Type{T}, ::Val{NT}, I::Vararg{Integer,N}) where {T, NT, N} = Tensor2DxyArray(Tensor2DxyArray(T, Val{NT-1}(), I...),Tensor2DxyArray(T, Val{NT-1}(), I...))

Tensor2DxzArray(a::AbstractArray{T,N},b::AbstractArray{T,N}) where {T,N} = TensorArray(a, ZeroTensorArray(tensor_ndims(T), size(a)...), b)
Tensor2DxzArray(::Type{T}, ::Val{1}, I::Vararg{Integer,N}) where {T, N} = Tensor2DxzArray(Array{T}(undef, I...), Array{T}(undef, I...))
Tensor2DxzArray(::Type{T}, ::Val{NT}, I::Vararg{Integer,N}) where {T, NT, N} = Tensor2DxzArray(Tensor2DxzArray(T, Val{NT-1}(), I...),Tensor2DxzArray(T, Val{NT-1}(), I...))

Tensor2DyzArray(a::AbstractArray{T,N},b::AbstractArray{T,N}) where {T,N} = TensorArray(ZeroTensorArray(tensor_ndims(T), size(a)...), a, b)
Tensor2DyzArray(::Type{T}, ::Val{1}, I::Vararg{Integer,N}) where {T, N} = Tensor2DyzArray(Array{T}(undef, I...), Array{T}(undef, I...))
Tensor2DyzArray(::Type{T}, ::Val{NT}, I::Vararg{Integer,N}) where {T, NT, N} = Tensor2DyzArray(Tensor2DyzArray(T, Val{NT-1}(), I...),Tensor2DyzArray(T, Val{NT-1}(), I...))

Tensor1DxArray(a::AbstractArray{T,N}) where {T,N} = TensorArray(a, ZeroTensorArray(tensor_ndims(T), size(a)...), ZeroTensorArray(tensor_ndims(T), size(a)...))
Tensor1DxArray(::Type{T}, ::Val{1}, I::Vararg{Integer,N}) where {T, N} = Tensor1DxArray(Array{T}(undef, I...))
Tensor1DxArray(::Type{T}, ::Val{NT}, I::Vararg{Integer,N}) where {T, NT, N} = Tensor1DxArray(Tensor1DxArray(T, Val{NT-1}(), I...))

Tensor1DyArray(a::AbstractArray{T,N}) where {T,N} = TensorArray(ZeroTensorArray(tensor_ndims(T), size(a)...), a, ZeroTensorArray(tensor_ndims(T), size(a)...))
Tensor1DyArray(::Type{T}, ::Val{1}, I::Vararg{Integer,N}) where {T, N} = Tensor1DyArray(Array{T}(undef, I...))
Tensor1DyArray(::Type{T}, ::Val{NT}, I::Vararg{Integer,N}) where {T, NT, N} = Tensor1DyArray(Tensor1DyArray(T, Val{NT-1}(), I...))

Tensor1DzArray(a::AbstractArray{T,N}) where {T,N} = TensorArray(ZeroTensorArray(tensor_ndims(T), size(a)...), ZeroTensorArray(tensor_ndims(T), size(a)...), a)
Tensor1DzArray(::Type{T}, ::Val{1}, I::Vararg{Integer,N}) where {T, N} = Tensor1DzArray(Array{T}(undef, I...))
Tensor1DzArray(::Type{T}, ::Val{NT}, I::Vararg{Integer,N}) where {T, NT, N} = Tensor1DzArray(Tensor1DzArray(T, Val{NT-1}(), I...))


function VecArray(a::Union{<:AbstractArray{Ta,N},Zero}, b::Union{<:AbstractArray{Tb,N},Zero}, c::Union{<:AbstractArray{Tc,N},Zero}) where {Ta,Tb,Tc,N}

    if (eltype(a)<:AbstractTensor || eltype(b)<:AbstractTensor || eltype(c)<:AbstractTensor)
        throw(ArgumentError("Array of Tensors are not valid input to the `VecArray` function"))
    end

    return TensorArray(x=a, y=b, z=c)
end

VecArray(;x = 𝟎, y = 𝟎, z = 𝟎) = VecArray(x,y,z)

Vec3DArray(a::AbstractArray{T,N}, b::AbstractArray{T,N}, c::AbstractArray{T,N}) where {T,N} = VecArray(a, b, c)
Vec3DArray{T}(I::Vararg{Integer,N}) where {T,N} = Vec3DArray(Array{T}(undef,I...), Array{T}(undef,I...), Array{T}(undef,I...))
VecArray{T}(I::Vararg{Integer,N}) where {T,N} = Vec3DArray{T}(I...)

Vec2DxyArray(a::AbstractArray{T,N},b::AbstractArray{T,N}) where {T,N} = VecArray(x=a, y=b)
Vec2DxyArray{T}(I::Vararg{Integer,N}) where {T,N} = Tensor2DxyArray(T,Val{1}(),I...)

Vec2DxzArray(a::AbstractArray{T,N},b::AbstractArray{T,N}) where {T,N} = VecArray(x=a, z=b)
Vec2DxzArray{T}(I::Vararg{Integer,N}) where {T,N} = Tensor2DxzArray(T,Val{1}(),I...)

Vec2DyzArray(a::AbstractArray{T,N},b::AbstractArray{T,N}) where {T,N} = VecArray(y=a, z=b)
Vec2DyzArray{T}(I::Vararg{Integer,N}) where {T,N} = Tensor2DyzArray(T,Val{1}(),I...)

Vec1DxArray(a::AbstractArray{T,N}) where {T,N} = VecArray(x=a)
Vec1DxArray{T}(I::Vararg{Integer,N}) where {T,N} = Tensor1DxArray(T,Val{1}(),I...)

Vec1DyArray(a::AbstractArray{T,N}) where {T,N} = VecArray(y=a)
Vec1DyArray{T}(I::Vararg{Integer,N}) where {T,N} = Tensor1DyArray(T,Val{1}(),I...)

Vec1DzArray(a::AbstractArray{T,N}) where {T,N} = VecArray(z=a)
Vec1DzArray{T}(I::Vararg{Integer,N}) where {T,N} = Tensor1DzArray(T,Val{1}(),I...)

#Resolve ambiguities
Vec1DxArray{Zero}(I::Vararg{Integer,N}) where {N} = Tensor1DxArray(Zero,Val{1}(),I...)

Ten3DArray{T}(I::Vararg{Integer,N}) where {T,N} = TensorArray{T,2}(I...)
TenArray{T}(I::Vararg{Integer,N}) where {T,N} = Ten3DArray{T}(I...)

Ten2DxyArray(a::AbstractArray{T,N},b::AbstractArray{T,N}) where {T<:Vec2Dxy,N} = Tensor2DxyArray(a, b)
Ten2DxyArray{T}(I::Vararg{Integer,N}) where {T,N} = Tensor2DxyArray(T,Val{2}(),I...)

Ten2DxzArray(a::AbstractArray{T,N},b::AbstractArray{T,N}) where {T<:Vec2Dxz,N} = Tensor2DxzArray(a, b)
Ten2DxzArray{T}(I::Vararg{Integer,N}) where {T,N} = Tensor2DxzArray(T,Val{2}(),I...)

Ten2DyzArray(a::AbstractArray{T,N},b::AbstractArray{T,N}) where {T<:Vec2Dyz,N} = Tensor2DyzArray(a, b)
Ten2DyzArray{T}(I::Vararg{Integer,N}) where {T,N} = Tensor2DyzArray(T,Val{2}(),I...)

Ten1DxArray(a::AbstractArray{T,N}) where {T<:Vec1Dx,N} = Tensor1DxArray(a)
Ten1DxArray{T}(I::Vararg{Integer,N}) where {T,N} = Tensor1DxArray(T,Val{2}(),I...)

Ten1DyArray(a::AbstractArray{T,N}) where {T<:Vec1Dy,N} = Tensor1DyArray(a)
Ten1DyArray{T}(I::Vararg{Integer,N}) where {T,N} = Tensor1DyArray(T,Val{2}(),I...)

Ten1DzArray(a::AbstractArray{T,N}) where {T<:Vec1Dz,N} = Tensor1DzArray(a)
Ten1DzArray{T}(I::Vararg{Integer,N}) where {T,N} = Tensor1DzArray(T,Val{2}(),I...)

#Resolve ambiguities
Ten1DxArray{Zero}(I::Vararg{Integer,N}) where {N} = Tensor1DxArray(Zero,Val{2}(),I...)

####################################


######################### Special Second Order Tensor constructors ###############################


@inline function TenArray(xx::AbstractArray{Txx,N}, xy::AbstractArray{Txy,N}, xz::AbstractArray{Txz,N},
                          yx::AbstractArray{Tyx,N}, yy::AbstractArray{Tyy,N}, yz::AbstractArray{Tyz,N},
                          zx::AbstractArray{Tzx,N}, zy::AbstractArray{Tzy,N}, zz::AbstractArray{Tzz,N}) where {Txx,Txy,Txz,
                                                                                                               Tyx,Tyy,Tyz,
                                                                                                               Tzx,Tzy,Tzz,N}
    x = VecArray(xx, yx, zx)
    y = VecArray(xy, yy, zy)
    z = VecArray(xz, yz, zz)
    return TensorArray(x, y, z)
end

_if_zero_to_Array(s::NTuple{N, Int}, ::Zero) where {N} = ZeroArray(s)
_if_zero_to_Array(::NTuple{N, Int}, x::AbstractArray) where {N} = x

function TenArray(;
        xx = 𝟎, xy = 𝟎, xz = 𝟎,
        yx = 𝟎, yy = 𝟎, yz = 𝟎,
        zx = 𝟎, zy = 𝟎, zz = 𝟎,
    )

    vals = (
        xx, xy, xz,
        yx, yy, yz,
        zx, zy, zz,
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

    return TenArray(final_vals...)
end


Ten2DxyArray(xx::AbstractArray, xy::AbstractArray, yx::AbstractArray, yy::AbstractArray) = TenArray(xx=xx,xy=xy,yx=yx,yy=yy)
Ten2DxzArray(xx::AbstractArray, xz::AbstractArray, zx::AbstractArray, zz::AbstractArray) = TenArray(xx=xx,xz=xz,zx=zx,zz=zz)
Ten2DyzArray(yy::AbstractArray, yz::AbstractArray, zy::AbstractArray, zz::AbstractArray) = TenArray(yy=yy,yz=yz,zy=zy,zz=zz)
Ten1DxArray(xx::AbstractArray) = TenArray(xx=xx)
Ten1DyArray(yy::AbstractArray) = TenArray(yy=yy)
Ten1DzArray(zz::AbstractArray) = TenArray(zz=zz)

########################################################################

################# AbstractArray interface and other ####################

@inline Base.size(A::AbstractTensorArray) = size(getfield(A,1))
@inline Base.length(A::AbstractTensorArray) = length(getfield(A,1))

Base.dataids(A::TensorArray) = (Base.dataids(A.x)..., Base.dataids(A.y)..., Base.dataids(A.z)...) 

struct IndexHelper{N}
    i::NTuple{N,Int}
end

@inline (s::IndexHelper{N})(A) where {N} = @inline @inbounds(Base.getindex(A,(s.i)...))
@inline (s::IndexHelper{N})(A,v) where {N} = @inline @inbounds(Base.setindex!(A,v,(s.i)...))

@inline function Base.getindex(A::AbstractTensorArray{T}, i::Int) where T<:AbstractTensor
    @boundscheck checkbounds(A, i)
    @inline r = constructor(T)(map(IndexHelper((i,)),fields(A))...)
    return r
end

@inline function Base.getindex(A::AbstractTensorArray{T, N}, I::Vararg{Int, N}) where {T, N}
    @boundscheck checkbounds(A, I...)
    @inline r = constructor(T)(map(IndexHelper(I),fields(A))...)
    return r
end

@inline function Base.setindex!(A::AbstractTensorArray{T}, u::T2, i::Int) where {N,T<:AbstractTensor{N}, T2<:AbstractTensor{N}}
    @boundscheck checkbounds(A, i)
    uc = convert(T, u)
    @inline map(IndexHelper((i,)), fields(A), fields(uc))
    return A
end

@inline function Base.setindex!(A::AbstractTensorArray{T, N}, u::T2, I::Vararg{Int, N}) where {N, NT, T<:AbstractTensor{NT}, T2<:AbstractTensor{NT}}
    @boundscheck checkbounds(A, I...)
    uc = convert(T, u)
    @inline map(IndexHelper(I), fields(A), fields(uc))
    return A
end

Base.resize!(A::AbstractTensorArray{<:Any, 1}, i::Integer) = begin map(x -> resize!(x, i), fields(A)) ; A end

Base.similar(A::TensorArray, ::Type{Tensor{N, Tt, Tx, Ty, Tz}}, dims::Tuple{Int, Vararg{Int, N2}}) where {Tt, N, Tx, Ty, Tz, N2} = TensorArray(similar(A.x, Tx, dims), similar(A.y, Ty, dims), similar(A.z, Tz, dims))

function tensorarray(::Type{Tensor{1,T,Tx,Ty,Tz}}, dims::Dims) where {T,Tx,Ty,Tz}
    return TensorArray(Array{Tx}(undef,dims), Array{Ty}(undef,dims), Array{Tz}(undef,dims))
end

function tensorarray(::Type{Tensor{N,T,Tx,Ty,Tz}}, dims::Dims) where {N,T,Tx,Ty,Tz}
    return TensorArray(tensorarray(Tx,dims), tensorarray(Ty,dims), tensorarray(Tz,dims))
end

tensorarray(::Type{T}, d::Integer, dims::Vararg{Integer}) where {T<:AbstractTensor} = tensorarray(T,Dims((d,dims...)))

#Definitons so broadcast return a VecArray =======================================

function Base.similar(bc::Broadcast.Broadcasted, ::Type{T}) where {T<:AbstractTensor}
    s = length.(axes(bc))
    return tensorarray(T,s)
end

@inline function Base.getproperty(T::TenArray, s::Symbol)
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

