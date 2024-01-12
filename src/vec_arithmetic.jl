import Base: +, -, *, /, //, muladd, conj, ==
import LinearAlgebra: dot, ⋅, norm, cross, ×, normalize
export dot, ⋅, norm, cross, ×, normalize # Reexport from LinearAlgebra

include("muladd_definitions.jl")

@inline +(a::AbstractVec) = a
@inline *(b::Number, v::AbstractVec) = Vec(x=b*v.x, y=b*v.y, z=b*v.z)
@inline *(v::AbstractVec, b::Number) = b*v
@inline /(v::AbstractVec,b::Number) = inv(b)*v
@inline //(v::AbstractVec,b::Number) = (One()//b)*v
@inline +(a::AbstractVec, b::AbstractVec) = Vec(x=a.x+b.x, y=a.y+b.y, z=a.z+b.z)
@inline +(a::AbstractVec...) = Vec(x=+(_x.(a)...), y=+(_y.(a)...), z=+(_z.(a)...))
@inline -(a::AbstractVec, b::AbstractVec) = Vec(x=a.x-b.x, y=a.y-b.y, z=a.z-b.z)
@inline -(a::AbstractVec) = _zero(a) - a
@inline ==(a::AbstractVec,b::AbstractVec) = a.x == b.x && a.y == b.y && a.z == b.z

@inline _muladd(a::Number, v::AbstractVec, u::AbstractVec) = Vec(x=_muladd(a,_x(v),_x(u)), y=_muladd(a,_y(v),_y(u)), z=_muladd(a,_z(v),_z(u)))
@inline _muladd(v::AbstractVec, a::Number, u::AbstractVec) = Vec(x=_muladd(a,_x(v),_x(u)), y=_muladd(a,_y(v),_y(u)), z=_muladd(a,_z(v),_z(u)))

@inline muladd(a::Number, v::AbstractVec, u::AbstractVec) = _muladd(a,v,u)
@inline muladd(v::AbstractVec, a::Number, u::AbstractVec) = _muladd(v,a,u)

@inline dot(a::AbstractVec,b::AbstractVec) = _muladd(a.x, b.x, _muladd(a.y, b.y, a.z*b.z))

@inline fsqrt(x) = @fastmath sqrt(x)

@inline conj(u::Vec) = Vec(x=conj(u.x), y=conj(u.y), z=conj(u.z))

@inline inner(u::Vec{T,1},v::Vec{T2,1}) where {T,T2} = dot(conj(u),v)

@inline norm(u::AbstractVec) = fsqrt(real(inner(u,u)))

@inline norm(u::Vec1Dx,p::Real=2) = abs(u.x)
@inline norm(u::Vec1Dy,p::Real=2) = abs(u.y)
@inline norm(u::Vec1Dz,p::Real=2) = abs(u.z)
@inline norm(u::Vec0D,p::Real=2) = 𝟎

@inline normalize(u::AbstractVec) = u/norm(u)
@inline normalize(u::AbstractVec{Zero}) = u

@inline function cross(a::AbstractVec,b::AbstractVec) 
    ax = a.x
    ay = a.y
    az = a.z
    bx = b.x
    by = b.y
    bz = b.z
    return  Vec(x=_muladd(ay, bz, -az*by), y=_muladd(az,bx, -ax*bz), z=_muladd(ax,by, -ay*bx))
end
