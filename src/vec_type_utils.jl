@inline _eltype_convert(::Type{T}, number) where {T} = _my_convert(T, number)

@inline function _eltype_convert(::Type{T}, tensor::TT) where {T, TT<:AbstractTensor}
    constructor(TT)(map(x -> _eltype_convert(T, x), fields(tensor))...)
end

"""
    tensor_type_3D(::Val{N}, T::DataType) -> Type{Tensor{N,T,Tx,Ty,Tz}} where {Tx,Ty,Tz}

Return the type of a 3D `N`th order tensor with element type of `T`.
"""
tensor_type_3D(::Val{1},::Type{T}) where T = Tensor{1,T,T,T,T}
function tensor_type_3D(::Val{N},::Type{T}) where {N,T}
    lTt = tensor_type_3D(Val{N-1}(),T)
    return Tensor{N,T,lTt,lTt,lTt}
end

"""
    tensor_type_2Dxy(::Val{N}, T::DataType) -> Type{Tensor{N,Union{T,Zero},Tx,Ty,Tz}} where {Tx,Ty,Tz}

Return the type of a 2D in xy `N`th order tensor with element type of `T`.
"""
tensor_type_2Dxy(::Val{1},::Type{T}) where T = Tensor{1,Union{T,Zero},T,T,Zero}
function tensor_type_2Dxy(::Val{N},::Type{T}) where {N,T}
    VNm1 = Val{N-1}()
    lTt = tensor_type_2Dxy(VNm1,T)
    lTzero = tensor_type_3D(VNm1,Zero)
    return Tensor{N,Union{T,Zero},lTt,lTt,lTzero}
end

"""
    tensor_type_2Dxz(::Val{N}, T::DataType) -> Type{Tensor{Union{T,Zero},N,Tx,Ty,Tz}} where {Tx,Ty,Tz}

Return the type of a 2D in xz `N`th order tensor with element type of `T`.
"""
tensor_type_2Dxz(::Val{1},::Type{T}) where T = Tensor{1,Union{T,Zero},T,Zero,T}
function tensor_type_2Dxz(::Val{N},::Type{T}) where {N,T}
    VNm1 = Val{N-1}()
    lTt = tensor_type_2Dxz(VNm1,T)
    lTzero = tensor_type_3D(VNm1,Zero)
    return Tensor{N,Union{T,Zero},lTt,lTzero,lTt}
end

"""
    tensor_type_2Dyz(::Val{N}, T::DataType) -> Type{Tensor{N,Union{T,Zero},Tx,Ty,Tz}} where {Tx,Ty,Tz}

Return the type of a 2D in yz `N`th order tensor with element type of `T`.
"""
tensor_type_2Dyz(::Val{1},::Type{T}) where T = Tensor{1,Union{T,Zero},Zero,T,T}
function tensor_type_2Dyz(::Val{N},::Type{T}) where {N,T}
    VNm1 = Val{N-1}()
    lTt = tensor_type_2Dyz(VNm1,T)
    lTzero = tensor_type_3D(VNm1,Zero)
    return Tensor{N,Union{T,Zero},lTzero,lTt,lTt}
end

"""
    tensor_type_1Dx(::Val{N}, T::DataType) -> Type{Tensor{N,Union{T,Zero},Tx,Ty,Tz}} where {Tx,Ty,Tz}

Return the type of a 1D in x `N`th order tensor with element type of `T`.
"""
tensor_type_1Dx(::Val{1},::Type{T}) where T = Tensor{1,Union{T,Zero},T,Zero,Zero}
function tensor_type_1Dx(::Val{N},::Type{T}) where {N,T}
    VNm1 = Val{N-1}()
    lTt = tensor_type_1Dx(VNm1,T)
    lTzero = tensor_type_3D(VNm1,Zero)
    return Tensor{N,Union{T,Zero},lTt,lTzero,lTzero}
end

"""
    tensor_type_1Dy(::Val{N}, T::DataType) -> Type{Tensor{N,Union{T,Zero},Tx,Ty,Tz}} where {Tx,Ty,Tz}

Return the type of a 1D in y `N`th order tensor with element type of `T`.
"""
tensor_type_1Dy(::Val{1},::Type{T}) where T = Tensor{1,Union{T,Zero},Zero,T,Zero}
function tensor_type_1Dy(::Val{N},::Type{T}) where {N,T}
    VNm1 = Val{N-1}()
    lTt = tensor_type_1Dy(VNm1,T)
    lTzero = tensor_type_3D(VNm1,Zero)
    return Tensor{N,Union{T,Zero},lTzero,lTt,lTzero}
end

"""
    tensor_type_1Dz(::Val{N}, T::DataType) -> Type{Tensor{N,Union{T,Zero},Tx,Ty,Tz}} where {Tx,Ty,Tz}

Return the type of a 1D in z `N`th order tensor with element type of `T`.
"""
tensor_type_1Dz(::Val{1},::Type{T}) where T = Tensor{1,Union{T,Zero},Zero,Zero,T}
function tensor_type_1Dz(::Val{N},::Type{T}) where {N,T}
    VNm1 = Val{N-1}()
    lTt = tensor_type_1Dz(VNm1,T)
    lTzero = tensor_type_3D(VNm1,Zero)
    return Tensor{N,Union{T,Zero},lTzero,lTzero,lTt}
end

@inline check_args_ignoring_zeros(::Vararg{Any,N2}) where {N2} = nothing

@inline function check_args_ignoring_zeros(::Union{<:AbstractTensor{N},Zero},::Union{<:AbstractTensor{N},Zero},::Union{<:AbstractTensor{N},Zero}) where {N}
    return nothing
end

@inline function check_args_ignoring_zeros(::Union{<:AbstractTensor{N1},Zero},::Union{<:AbstractTensor{N2},Zero},::Union{<:AbstractTensor{N3},Zero}) where {N1,N2,N3}
    return throw(DimensionMismatch())
end

@inline function check_args_ignoring_zeros(::Union{<:AbstractTensor{N},Zero},::Union{<:AbstractTensor{N},Zero},::Union{<:AbstractTensor{N},Zero},::Union{<:AbstractTensor{N},Zero},::Union{<:AbstractTensor{N},Zero},::Union{<:AbstractTensor{N},Zero}) where {N}
    return nothing
end

@inline function check_args_ignoring_zeros(::Union{<:AbstractTensor{N1},Zero},::Union{<:AbstractTensor{N2},Zero},::Union{<:AbstractTensor{N3},Zero},::Union{<:AbstractTensor{N4},Zero},::Union{<:AbstractTensor{N5},Zero},::Union{<:AbstractTensor{N6},Zero}) where {N1,N2,N3,N4,N5,N6}
    return throw(DimensionMismatch())
end

@inline _get_ndims(::Vararg{Any}) = Val{0}()
@inline function _get_ndims(::Vararg{Union{<:AbstractTensor{N},Zero}}) where {N}
    return Val{N}()
end

@inline if_zero_to_tensor(::Val{0}, ::Zero) = Zero()
@inline if_zero_to_tensor(::Val{N}, ::Zero) where {N} = Tensor{N}()
@inline if_zero_to_tensor(::Val{N}, x) where {N} = x
@inline if_zero_to_tensor(v::Val{N},var::Vararg) where {N} = map(x->if_zero_to_tensor(v,x), var)

