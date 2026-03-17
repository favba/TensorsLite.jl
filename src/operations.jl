
# define my own *, +, - so I can extend those operators without commiting type piracy (For SIMDExt.jl)
using Zeros: StaticBool
# I'm also using my own `dot` function, and LinearAlgebra.dot is overloaded in ext/LinearAlgebraExt.jl
@inline +(a) = Base.:+(a)
@inline +(a, b) = Base.:+(a, b)
@inline +(a::Vararg{<:Any,N}) where {N} = (+(a[Base.OneTo(N-1)]...)) + a[N] 
@inline -(a) = Base.:-(a)
@inline -(a, b) = Base.:-(a, b)
@inline *(a, b) = Base.:*(a, b)
@inline *(a::Vararg{<:Any,N}) where {N} = (*(a[Base.OneTo(N-1)]...)) * a[N] 

# `Zero`s will be the mark of a nill direction of a vector

@inline +(x, ::Zero) = x
@inline +(::Zero, x) = x
@inline +(::Zero, x::Zero) = x

##Useful for debuging only, should never happen
#@inline +(::AbstractTensor, ::Zero) = throw(DimensionMismatch("Cannot add Tensor and scalar"))
#@inline +(::Zero, ::AbstractTensor) = throw(DimensionMismatch("Cannot add Tensor and scalar"))
@inline +(x, ::One) = x + oneunit(x)
@inline +(::One, x) = oneunit(x) + x
@inline +(::One, ::One) = 2
##Useful for debuging only, should never happen
#@inline +(::AbstractTensor, ::One) = throw(DimensionMismatch("Cannot add Tensor and scalar"))
#@inline +(::One, ::AbstractTensor) = throw(DimensionMismatch("Cannot add Tensor and scalar"))
@inline +(::One, ::Zero) = One()
@inline +(::Zero, ::One) = One()

@inline -(x, ::Zero) = x
@inline -(::Zero, x) = -x
@inline -(::Zero, x::Zero) = x
##Useful for debuging only, should never happen
#@inline -(::AbstractTensor, ::Zero) = throw(DimensionMismatch("Cannot subtract Tensor and scalar"))
#@inline -(::Zero, ::AbstractTensor) = throw(DimensionMismatch("Cannot subtract Tensor and scalar"))
@inline -(x, ::One) = x - oneunit(x)
@inline -(::One, x) = oneunit(x) - x
@inline -(::One, x::One) = Zero()
##Useful for debuging only, should never happen
#@inline -(::AbstractTensor, ::One) = throw(DimensionMismatch("Cannot subtract Tensor and scalar"))
#@inline -(::One, ::AbstractTensor) = throw(DimensionMismatch("Cannot subtract Tensor and scalar"))
@inline -(::One, ::Zero) = One()
@inline -(::Zero, ::One) = -1

@inline *(::Any, ::Zero) = Zero()
@inline *(::Zero, ::Any) = Zero()
@inline *(x, ::One) = x
@inline *(::One, x) = x
@inline *(::Zero, ::Zero) = Zero()
@inline *(::Zero, ::One) = Zero()
@inline *(::One, ::Zero) = Zero()
@inline *(::One, ::One) = One()
@inline *(::Zero, v::T) where {T<:AbstractTensor} = constructor(T)(
    map(
        *,
        ntuple(
               i->Zero(),
               Val(fieldcount(T))
        ),
        fields(v) 
    )...
)
@inline *(v::T, ::Zero) where {T<:AbstractTensor} = Zero() * v
@inline *(::One, v::T) where {T<:AbstractTensor} = v
@inline *(v::T, ::One) where {T<:AbstractTensor} = v

#Using generated is easier than dealing with all the amibiguities...
@inline @generated function __muladd(::Type{T1}, ::Type{T2}, ::Type{T3}, a, b, c) where {T1,T2,T3}
    if (T1<:StaticBool || T2<:StaticBool || T3<:StaticBool) 
        :(return a * b + c) # `*` and `+` as defined in this module
    else
        :(return Base.muladd(a,b,c))
    end
end

@inline function _muladd(a, b, c)
    @inline __muladd(typeof(a),typeof(b),typeof(c),a,b,c)
end

############# Tensor Algebra ################

@inline Base.:+(a::AbstractTensor) = a
@inline Base.:+(a::AbstractTensor{N}, b::AbstractTensor{N}) where {N} = Tensor(a.x + b.x, a.y + b.y, a.z + b.z)

@inline Base.:+(a::AbstractTensor{N}...) where {N} = Tensor(+(map(_x, a)...), +(map(_y, a)...), +(map(_z, a)...))

# We treat Vec's as scalar for broadcasting but the default definition of + and - for AbstractArray's relies
# on broadcasting to perform addition and subtraction. The method definitons below overcomes this inconsistency
@inline Base.:+(a::AbstractTensor, b::AbstractArray) = Array(a) + b
@inline Base.:+(b::AbstractArray, a::AbstractTensor) = b + Array(a)

@inline Base.:-(a::T) where {T <: AbstractTensor} = @inline constructor(T)(map(-, fields(a))...)
@inline Base.:-(a::AbstractTensor{N}, b::AbstractTensor{N}) where {N} = Tensor(a.x - b.x, a.y - b.y, a.z - b.z)

@inline Base.:-(a::AbstractTensor, b::AbstractArray) = Array(a) - b
@inline Base.:-(b::AbstractArray, a::AbstractTensor) = b - Array(a)

import Base: ==
@inline (==)(a::AbstractTensor, b::AbstractTensor) = ((a.x == b.x) & (a.y == b.y) & (a.z == b.z))

# `b` is part of some ring, more general than Number
@inline *(b, v::T) where {T <: AbstractTensor} = @inline  begin
    bt = convert(promote_type(typeof(b), nonzero_eltype(T)), b)
    constructor(T)(map(*, ntuple(i -> bt, Val(fieldcount(T))), fields(v))...)
end
@inline Base.:*(b, v::AbstractTensor) = b*v
@inline Base.:*(b::Number, v::AbstractTensor) = b*v

# `b` is part of some ring, more general than Number
@inline *(v::AbstractTensor, b) = b * v
@inline Base.:*(v::AbstractTensor, b) = b * v
@inline Base.:*(v::AbstractTensor, b::Number) = b * v

@inline _muladd(a::Number, v::AbstractTensor{N}, u::AbstractTensor{N}) where {N} = Tensor(_muladd(a, v.x, u.x), _muladd(a, v.y, u.y), _muladd(a, v.z, u.z))

@inline _muladd(::Zero, ::AbstractTensor{N}, u::AbstractTensor{N}) where {N} = u

@inline _muladd(v::AbstractTensor{N}, a::Number, u::AbstractTensor{N}) where {N} = _muladd(a, v, u)
@inline _muladd(v::AbstractTensor{N}, a::Zero, u::AbstractTensor{N}) where {N} = _muladd(a, v, u)

@inline Base.muladd(a::Number, v::AbstractTensor{N}, u::AbstractTensor{N}) where {N} = _muladd(a,v,u)

@inline Base.muladd(v::AbstractTensor{N}, a::Number, u::AbstractTensor{N}) where {N} = Base.muladd(a, v, u)

@inline Base.:/(v::T, b::Number) where {T <: AbstractTensor} = @inline  begin
    ib = inv(b)
    bt = convert(promote_type(typeof(ib), nonzero_eltype(T)), ib)
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

@inline _muladd(a::Vec, b::Vec, c) = _muladd(a.x, b.x, _muladd(a.y, b.y, _muladd(a.z, b.z, c)))
@inline _muladd(a::Vec, b::Vec, c::StaticBool) = _muladd(a.x, b.x, _muladd(a.y, b.y, _muladd(a.z, b.z, c)))

@inline function _muladd(a::AbstractTensor{N1}, b::Vec, c::AbstractTensor{N2}) where {N1, N2}
    ((N1-1) === N2) || throw(DimensionMismatch())
    return _muladd(a.x, b.x, _muladd(a.y, b.y, _muladd(a.z, b.z, c)))
end
@inline function _muladd(a::AbstractTensor{N1}, b::AbstractTensor{N2}, c::AbstractTensor{N3}) where {N1, N2, N3}
    ((N1+N2-2) === N3) || throw(DimensionMismatch())
    return Tensor(_muladd(a, b.x, c.x), _muladd(a, b.y, c.y), _muladd(a, b.z, c.z))
end

@inline Base.muladd(a::AbstractTensor, b::AbstractTensor, c::AbstractTensor) = _muladd(a,b,c)

@inline dot(a::AbstractTensor,b::Vec) = _muladd(a.x, b.x, _muladd(a.y, b.y, a.z*b.z))
@inline dot(A::AbstractTensor, B::AbstractTensor) = Tensor(dot(A,B.x), dot(A, B.y), dot(A, B.z)) 
@inline Base.:*(T::AbstractTensor, B::AbstractTensor) = dot(T,B)

@inline dotadd(a::Vec,b::Vec,c) = _muladd(a,b,c)
@inline dotadd(a::AbstractTensor,b::AbstractTensor,c::AbstractTensor) = _muladd(a,b,c)

@inline inner(u::Vec, v::Vec) = dot(conj(u), v)

@inline inneradd(u::Vec, v::Vec, c) = dotadd(conj(u), v, c)

@inline inneradd(T1::AbstractTensor{N}, T2::AbstractTensor{N}, c) where {N} = inneradd(T1.x, T2.x, inneradd(T1.y, T2.y, inneradd(T1.z, T2.z, c)))

@inline inner(T1::AbstractTensor{N}, T2::AbstractTensor{N}) where {N} = inneradd(T1.x, T2.x, inneradd(T1.y, T2.y, inner(T1.z,T2.z)))

@inline dcontract(a::AbstractTensor,b::Ten) = _muladd(a.xx, b.xx, _muladd(a.xy, b.xy, _muladd(a.xz, b.xz,
                                              _muladd(a.yx, b.yx, _muladd(a.yy, b.yy, _muladd(a.yz, b.yz,
                                              _muladd(a.zx, b.zx, _muladd(a.zy, b.zy, a.zz*b.zz))))))))

@inline dcontract(A::AbstractTensor, B::AbstractTensor) = Tensor(dcontract(A, B.x), dcontract(A, B.y), dcontract(A, B.z)) 

const ⊡ = dcontract

@inline dcontractadd(a::Ten,b::Ten, c) = _muladd(a.xx, b.xx, _muladd(a.xy, b.xy, _muladd(a.xz, b.xz,
                                         _muladd(a.yx, b.yx, _muladd(a.yy, b.yy, _muladd(a.yz, b.yz,
                                         _muladd(a.zx, b.zx, _muladd(a.zy, b.zy, _muladd(a.zz, b.zz, c)))))))))

@inline function dcontractadd(a::AbstractTensor{N},b::Ten, c::AbstractTensor{N2}) where {N,N2}
    ((N-2) === N2) || throw(DimensionMismatch())
    return _muladd(a.xx, b.xx, _muladd(a.xy, b.xy, _muladd(a.xz, b.xz,
           _muladd(a.yx, b.yx, _muladd(a.yy, b.yy, _muladd(a.yz, b.yz,
           _muladd(a.zx, b.zx, _muladd(a.zy, b.zy, _muladd(a.zz, b.zz, c)))))))))
end

@inline function dcontractadd(A::AbstractTensor{N}, B::AbstractTensor{N2}, C::AbstractTensor{N3}) where {N,N2,N3}
    ((N-2 + N2-2) === N3) || throw(DimensionMismatch())
    return Tensor(dcontractadd(A, B.x, C.x), dcontractadd(A, B.y, C.y), dcontractadd(A, B.z, C.z)) 
end
