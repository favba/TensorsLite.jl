@inline _eltype_convert(::Type{T}, number) where {T} = _my_convert(T, number)

@inline function _eltype_convert(::Type{T}, vec::AbstractTensor{Tv, N}) where {T, Tv, N}
    Tensor(_eltype_convert(T, vec.x), _eltype_convert(T, vec.y), _eltype_convert(T, vec.z))
end

"""
    tensor_type_3D(::Val{N}, T::DataType) -> Type{Tensor{T,N,Tx,Ty,Tz}} where {Tx,Ty,Tz}

Return the type of a 3D `N`th order tensor with element type of `T`.
"""
tensor_type_3D(::Val{1},::Type{T}) where T = Tensor{T,1,T,T,T}
function tensor_type_3D(::Val{N},::Type{T}) where {N,T}
    lTt = tensor_type_3D(Val{N-1}(),T)
    return Tensor{T,N,lTt,lTt,lTt}
end

"""
    tensor_type_2Dxy(::Val{N}, T::DataType) -> Type{Tensor{Union{T,Zero},N,Tx,Ty,Tz}} where {Tx,Ty,Tz}

Return the type of a 2D in xy `N`th order tensor with element type of `T`.
"""
tensor_type_2Dxy(::Val{1},::Type{T}) where T = Tensor{Union{T,Zero},1,T,T,Zero}
function tensor_type_2Dxy(::Val{N},::Type{T}) where {N,T}
    VNm1 = Val{N-1}()
    lTt = tensor_type_2Dxy(VNm1,T)
    lTzero = tensor_type_3D(VNm1,Zero)
    return Tensor{Union{T,Zero},N,lTt,lTt,lTzero}
end

"""
    tensor_type_2Dxz(::Val{N}, T::DataType) -> Type{Tensor{Union{T,Zero},N,Tx,Ty,Tz}} where {Tx,Ty,Tz}

Return the type of a 2D in xz `N`th order tensor with element type of `T`.
"""
tensor_type_2Dxz(::Val{1},::Type{T}) where T = Tensor{Union{T,Zero},1,T,Zero,T}
function tensor_type_2Dxz(::Val{N},::Type{T}) where {N,T}
    VNm1 = Val{N-1}()
    lTt = tensor_type_2Dxz(VNm1,T)
    lTzero = tensor_type_3D(VNm1,Zero)
    return Tensor{Union{T,Zero},N,lTt,lTzero,lTt}
end

"""
    tensor_type_2Dyz(::Val{N}, T::DataType) -> Type{Tensor{Union{T,Zero},N,Tx,Ty,Tz}} where {Tx,Ty,Tz}

Return the type of a 2D in yz `N`th order tensor with element type of `T`.
"""
tensor_type_2Dyz(::Val{1},::Type{T}) where T = Tensor{Union{T,Zero},1,Zero,T,T}
function tensor_type_2Dyz(::Val{N},::Type{T}) where {N,T}
    VNm1 = Val{N-1}()
    lTt = tensor_type_2Dyz(VNm1,T)
    lTzero = tensor_type_3D(VNm1,Zero)
    return Tensor{Union{T,Zero},N,lTzero,lTt,lTt}
end

"""
    tensor_type_1Dx(::Val{N}, T::DataType) -> Type{Tensor{Union{T,Zero},N,Tx,Ty,Tz}} where {Tx,Ty,Tz}

Return the type of a 1D in x `N`th order tensor with element type of `T`.
"""
tensor_type_1Dx(::Val{1},::Type{T}) where T = Tensor{Union{T,Zero},1,T,Zero,Zero}
function tensor_type_1Dx(::Val{N},::Type{T}) where {N,T}
    VNm1 = Val{N-1}()
    lTt = tensor_type_1Dx(VNm1,T)
    lTzero = tensor_type_3D(VNm1,Zero)
    return Tensor{Union{T,Zero},N,lTt,lTzero,lTzero}
end

"""
    tensor_type_1Dy(::Val{N}, T::DataType) -> Type{Tensor{Union{T,Zero},N,Tx,Ty,Tz}} where {Tx,Ty,Tz}

Return the type of a 1D in y `N`th order tensor with element type of `T`.
"""
tensor_type_1Dy(::Val{1},::Type{T}) where T = Tensor{Union{T,Zero},1,Zero,T,Zero}
function tensor_type_1Dy(::Val{N},::Type{T}) where {N,T}
    VNm1 = Val{N-1}()
    lTt = tensor_type_1Dy(VNm1,T)
    lTzero = tensor_type_3D(VNm1,Zero)
    return Tensor{Union{T,Zero},N,lTzero,lTt,lTzero}
end

"""
    tensor_type_1Dz(::Val{N}, T::DataType) -> Type{Tensor{Union{T,Zero},N,Tx,Ty,Tz}} where {Tx,Ty,Tz}

Return the type of a 1D in z `N`th order tensor with element type of `T`.
"""
tensor_type_1Dz(::Val{1},::Type{T}) where T = Tensor{Union{T,Zero},1,Zero,Zero,T}
function tensor_type_1Dz(::Val{N},::Type{T}) where {N,T}
    VNm1 = Val{N-1}()
    lTt = tensor_type_1Dz(VNm1,T)
    lTzero = tensor_type_3D(VNm1,Zero)
    return Tensor{Union{T,Zero},N,lTzero,lTzero,lTt}
end

