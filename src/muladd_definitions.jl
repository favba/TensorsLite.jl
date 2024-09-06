#Base definitions
@inline _muladd(x, y, z) = muladd(x, y, z)
@inline _muladd(::Zero, ::Zero, ::Zero) = Zero()
@inline _muladd(::Zero, ::Number, ::Zero) = Zero()
@inline _muladd(::Number, ::Zero, ::Zero) = Zero()
@inline _muladd(::Zero, y::Number, x::Number) = convert(promote_type(typeof(y), typeof(x)), x)
@inline _muladd(y::Number, ::Zero, x::Number) = convert(promote_type(typeof(y), typeof(x)), x)
@inline _muladd(::Zero, ::Zero, x::Number) = x
@inline _muladd(x::Number, y::Number, ::Zero) = x * y
@inline _muladd(::One, x::Number, y::Number) = x + y
@inline _muladd(x::Number, ::One, y::Number) = x + y

#Resolving Ambiguities
@inline _muladd(y::One, ::Zero, x::Number) = x
@inline _muladd(y::Zero, ::One, x::Number) = x

@inline _muladd(::One, ::Zero, ::Zero) = Zero()

@inline _muladd(::One, x::Number, y::Zero) = x

@inline _muladd(::Zero, ::One, ::Zero) = Zero()

@inline _muladd(x::Number, ::One, y::Zero) = x

@inline _muladd(::One, ::One, y::Number) = y + One()
@inline _muladd(::One, ::One, y::Zero) = One()
