
############# Tensor Algebra ################

@inline Base.:+(a::AbstractTensor) = a

@inline Base.:+(a::AbstractTensor{N}, b::AbstractTensor{N}) where {N} = Tensor(a.x + b.x, a.y + b.y, a.z + b.z)

@inline Base.:+(a::AbstractTensor{N}...) where {N} = Tensor(+(map(_x, a)...), +(map(_y, a)...), +(map(_z, a)...))

# We treat Vec's as scalar for broadcasting but the default definition of + and - for AbstractArray's relies
# on broadcasting to perform addition and subtraction. The method definitons below overcomes this inconsistency
@inline Base.:+(a::AbstractTensor{N}, b::AbstractArray{<:Any, N}) where {N} = Array(a) + b

@inline Base.:+(b::AbstractArray{<:Any, N}, a::AbstractTensor{N}) where {N} = b + Array(a)

@inline Base.:-(a::T) where {T <: AbstractTensor} = @inline constructor(T)(map(-, fields(a))...)

@inline Base.:-(a::AbstractTensor{N}, b::AbstractTensor{N}) where {N} = Tensor(a.x - b.x, a.y - b.y, a.z - b.z)

@inline Base.:-(a::AbstractTensor{N}, b::AbstractArray{<:Any, N}) where {N} = Array(a) - b

@inline Base.:-(b::AbstractArray{<:Any, N}, a::AbstractTensor{N}) where {N} = b - Array(a)

import Base: ==
@inline (==)(a::AbstractTensor{N}, b::AbstractTensor{N}) where {N} = ((a.x == b.x) & (a.y == b.y) & (a.z == b.z))

@inline Base.:*(b::Number, v::T) where {T <: AbstractTensor} = @inline  begin
    bt = _my_convert(promote_type(typeof(b), nonzero_eltype(T)), b)
    constructor(T)(map(*, ntuple(i -> bt, Val(fieldcount(T))), fields(v))...)
end

@inline Base.:*(v::AbstractTensor, b::Number) = b * v

@inline Base.muladd(b::Number, v::T, u::T) where {T <: AbstractTensor} = @inline  begin
    bt = _my_convert(promote_type(typeof(b), nonzero_eltype(T)), b)
    constructor(T)(map(muladd, ntuple(i -> bt, Val(fieldcount(T))), fields(v), fields(u))...)
end

@inline Base.muladd(a::Number, v::AbstractTensor{N}, u::AbstractTensor{N}) where {N} = Tensor(muladd(a, v.x, u.x), muladd(a, v.y, u.y), muladd(a, v.z, u.z))

@inline Base.muladd(::Zero, ::AbstractTensor{N}, u::AbstractTensor{N}) where {N} = u

@inline Base.muladd(::One, v::AbstractTensor{N}, u::AbstractTensor{N}) where {N} = v+u

@inline Base.muladd(v::AbstractTensor{N}, a::Number, u::AbstractTensor{N}) where {N} = muladd(a, v, u)

@inline Base.:/(v::T, b::Number) where {T <: AbstractTensor} = @inline  begin
    ib = inv(b)
    bt = _my_convert(promote_type(typeof(ib), nonzero_eltype(T)), ib)
    constructor(T)(map(*, fields(v), ntuple(i -> bt, Val(fieldcount(T))))...)
end

@inline Base.://(v::AbstractTensor, b::Number) = (One() // b) * v

@inline Base.zero(::Type{T}) where {T <: AbstractTensor} = @inline constructor(T)(_zero_for_tuple(fieldtypes(T)...)...)

@inline Base.zero(::T) where {T <: AbstractTensor} = zero(T)

@inline Base.conj(a::T) where {T <: AbstractTensor} = @inline constructor(T)(map(conj, fields(a))...)

@inline _otimes(a,b) = a*b

@inline _otimes(a::AbstractTensor, b::AbstractTensor) = Tensor(_otimes(a, b.x), _otimes(a, b.y), _otimes(a, b.z))

@inline otimes(a::AbstractTensor, b::AbstractTensor) = _otimes(a,b)

const ⊗ = otimes

@inline Base.muladd(a::Vec, b::Vec, c::Number) = muladd(a.x, b.x, muladd(a.y, b.y, muladd(a.z, b.z, c)))

@inline function Base.muladd(a::AbstractTensor{N1}, b::Vec, c::AbstractTensor{N2}) where {N1, N2}
    ((N1-1) === N2) || throw(DimensionMismatch())
    return muladd(a.x, b.x, muladd(a.y, b.y, muladd(a.z, b.z, c)))
end

@inline function Base.muladd(a::AbstractTensor{N1}, b::AbstractTensor{N2}, c::AbstractTensor{N3}) where {N1, N2, N3}
    ((N1+N2-2) === N3) || throw(DimensionMismatch())
    return Tensor(muladd(a, b.x, c.x), muladd(a, b.y, c.y), muladd(a, b.z, c.z))
end

@inline dot(a::AbstractTensor,b::Vec) = muladd(a.x, b.x, muladd(a.y, b.y, a.z*b.z))

@inline dot(A::AbstractTensor, B::AbstractTensor) = Tensor(dot(A,B.x), dot(A, B.y), dot(A, B.z)) 

@inline Base.:*(T::AbstractTensor, B::AbstractTensor) = dot(T,B)

@inline dotadd(a::Vec,b::Vec,c::Number) = muladd(a,b,c)

@inline dotadd(a::AbstractTensor,b::AbstractTensor,c::AbstractTensor) = muladd(a,b,c)

@inline inner(u::Vec, v::Vec) = dot(conj(u), v)

@inline inneradd(u::Vec, v::Vec, c::Number) = dotadd(conj(u), v, c)

@inline inneradd(T1::AbstractTensor{N}, T2::AbstractTensor{N}, c::Number) where {N} = inneradd(T1.x, T2.x, inneradd(T1.y, T2.y, inneradd(T1.z, T2.z, c)))

@inline inner(T1::AbstractTensor{N}, T2::AbstractTensor{N}) where {N} = inneradd(T1.x, T2.x, inneradd(T1.y, T2.y, inner(T1.z,T2.z)))

@inline dcontract(a::AbstractTensor,b::Ten) = muladd(a.xx, b.xx, muladd(a.xy, b.xy, muladd(a.xz, b.xz,
                                              muladd(a.yx, b.yx, muladd(a.yy, b.yy, muladd(a.yz, b.yz,
                                              muladd(a.zx, b.zx, muladd(a.zy, b.zy, a.zz*b.zz))))))))

@inline dcontract(A::AbstractTensor, B::AbstractTensor) = Tensor(dcontract(A, B.x), dcontract(A, B.y), dcontract(A, B.z)) 

const ⊡ = dcontract

@inline dcontractadd(a::Ten,b::Ten, c::Number) = muladd(a.xx, b.xx, muladd(a.xy, b.xy, muladd(a.xz, b.xz,
                                         muladd(a.yx, b.yx, muladd(a.yy, b.yy, muladd(a.yz, b.yz,
                                         muladd(a.zx, b.zx, muladd(a.zy, b.zy, muladd(a.zz, b.zz, c)))))))))

@inline function dcontractadd(a::AbstractTensor{N},b::Ten, c::AbstractTensor{N2}) where {N,N2}
    ((N-2) === N2) || throw(DimensionMismatch())
    return muladd(a.xx, b.xx, muladd(a.xy, b.xy, muladd(a.xz, b.xz,
           muladd(a.yx, b.yx, muladd(a.yy, b.yy, muladd(a.yz, b.yz,
           muladd(a.zx, b.zx, muladd(a.zy, b.zy, muladd(a.zz, b.zz, c)))))))))
end

@inline function dcontractadd(A::AbstractTensor{N}, B::AbstractTensor{N2}, C::AbstractTensor{N3}) where {N,N2,N3}
    ((N-2 + N2-2) === N3) || throw(DimensionMismatch())
    return Tensor(dcontractadd(A, B.x, C.x), dcontractadd(A, B.y, C.y), dcontractadd(A, B.z, C.z)) 
end
