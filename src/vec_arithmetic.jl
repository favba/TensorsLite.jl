import Base: +, -, *, /, //, fma, muladd
import LinearAlgebra: dot, norm, cross

#Copied from Zeros.jl source code. Was commented out for some reason
for op in [:fma :muladd]
    @eval $op(::Zero, ::Zero, ::Zero) = Zero()
    for T in (Real, Integer)
        @eval $op(::Zero, ::$T, ::Zero) = Zero()
        @eval $op(::$T, ::Zero, ::Zero) = Zero()
        @eval $op(::Zero, x::$T, y::$T) = convert(promote_type(typeof(x),typeof(y)),y)
        @eval $op(x::$T, ::Zero, y::$T) = convert(promote_type(typeof(x),typeof(y)),y)
        @eval $op(x::$T, y::$T, ::Zero) = x*y
        @eval $op(::One, x::$T, y::$T) = x+y
        @eval $op(x::$T, ::One, y::$T) = x+y
    end
end

muladd(::Zero,::One,::Zero) = Zero()
muladd(::Zero,::Zero,x::Real) = x
muladd(::Zero,::Zero,x::Integer) = x
muladd(x::Real,::One,::Zero) = x
muladd(x::Zero,::One,y::Integer) = y
muladd(::One,::One,y::Integer) = One() + y
muladd(::One,::Zero,y::Zero) = Zero()
muladd(::One,::One,y::Zero) = One()
muladd(::One,::Zero,y::Integer) = y
muladd(x::Integer,::One,y::Zero) = x

@inline +(a::AbstractVec) = a
@inline *(b::Number, v::AbstractVec) = Vec(b*v.x, b*v.y, b*v.z)
@inline *(v::AbstractVec, b::Number) = b*v
@inline /(v::AbstractVec,b::Number) = inv(b)*v
@inline //(v::AbstractVec,b::Number) = (One()//b)*v
@inline +(a::AbstractVec, b::AbstractVec) = Vec(a.x+b.x, a.y+b.y, a.z+b.z)
@inline +(a::AbstractVec...) = Vec(+(xx.(a)...), +(yy.(a)...), +(zz.(a)...))
@inline -(a::AbstractVec, b::AbstractVec) = Vec(a.x-b.x, a.y-b.y, a.z-b.z)
@inline -(a::AbstractVec) = Vec0D() - a

@inline muladd(a::Number, v::AbstractVec, u::AbstractVec) = Vec(muladd(a,xx(v),xx(u)), muladd(a,yy(v),yy(u)), muladd(a,zz(v),zz(u)))
@inline muladd(v::AbstractVec, a::Number, u::AbstractVec) = Vec(muladd(a,xx(v),xx(u)), muladd(a,yy(v),yy(u)), muladd(a,zz(v),zz(u)))

@inline dot(a::AbstractVec,b::AbstractVec) = @muladd a.x*b.x + a.y*b.y + a.z*b.z

@muladd @inline function cross(a::AbstractVec,b::AbstractVec) 
    ax = a.x
    ay = a.y
    az = a.z
    bx = b.x
    by = b.y
    bz = b.z
    return  Vec(ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx)
end
