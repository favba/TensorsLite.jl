import Base: +, -, *, /, //, fma, muladd, conj
import LinearAlgebra: dot, norm, cross, normalize

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
@inline *(b::Number, v::AbstractVec) = Vec(x=b*v.x, y=b*v.y, z=b*v.z)
@inline *(v::AbstractVec, b::Number) = b*v
@inline /(v::AbstractVec,b::Number) = inv(b)*v
@inline //(v::AbstractVec,b::Number) = (One()//b)*v
@inline +(a::AbstractVec, b::AbstractVec) = Vec(x=a.x+b.x, y=a.y+b.y, z=a.z+b.z)
@inline +(a::AbstractVec...) = Vec(x=+(_x.(a)...), y=+(_y.(a)...), z=+(_z.(a)...))
@inline -(a::AbstractVec, b::AbstractVec) = Vec(x=a.x-b.x, y=a.y-b.y, z=a.z-b.z)
@inline -(a::AbstractVec) = 𝟎⃗ - a
#
@inline muladd(a::Number, v::AbstractVec, u::AbstractVec) = Vec(x=muladd(a,_x(v),_x(u)), y=muladd(a,_y(v),_y(u)), z=muladd(a,_z(v),_z(u)))
@inline muladd(v::AbstractVec, a::Number, u::AbstractVec) = Vec(x=muladd(a,_x(v),_x(u)), y=muladd(a,_y(v),_y(u)), z=muladd(a,_z(v),_z(u)))
#
@inline dot(a::AbstractVec,b::AbstractVec) = @muladd a.x*b.x + a.y*b.y + a.z*b.z

@inline fsqrt(x) = @fastmath sqrt(x)

@inline conj(u::Vec) = Vec(x=conj(u.x), y=conj(u.y), z=conj(u.z))

@inline inner(u::Vec{T,1},v::Vec{T2,1}) where {T,T2} = dot(u,conj(v))

@inline norm(u::Vec{T,1}) where T = fsqrt(inner(u,u))

@inline norm(u::Vec{T1,1,T2,Zero,Zero},p=2) where {T1,T2} = abs(u.x)
@inline norm(u::Vec{T1,1,Zero,T2,Zero},p=2) where {T1,T2} = abs(u.y)
@inline norm(u::Vec{T1,1,Zero,Zero,T2},p=2) where {T1,T2} = abs(u.z)

@inline normalize(u::Vec) = u/norm(u)
#
@muladd @inline function cross(a::AbstractVec,b::AbstractVec) 
    ax = a.x
    ay = a.y
    az = a.z
    bx = b.x
    by = b.y
    bz = b.z
    return  Vec(x=ay*bz - az*by, y=az*bx - ax*bz, z=ax*by - ay*bx)
end
