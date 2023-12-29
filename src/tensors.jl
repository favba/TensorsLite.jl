import LinearAlgebra: transpose, adjoint

@inline function Ten(;xx=𝟎, yx=𝟎, zx=𝟎, xy=𝟎, yy=𝟎, zy=𝟎, xz=𝟎, yz=𝟎, zz=𝟎)
    x = Vec(xx,yx,zx)
    y = Vec(xy,yy,zy)
    z = Vec(xz,yz,zz)
    return Vec(x,y,z)
end

@inline *(T::AbstractTen,v::AbstractVec{<:Any,1}) = dot(T,v)
@inline *(v::AbstractVec{<:Any,1},T::AbstractTen) = dot(transpose(T),v)

@inline function transpose(T::AbstractTen)
    x = T.x
    y = T.y
    z = T.z
    nx = Vec(x.x,y.x,z.x)
    ny = Vec(x.y,y.y,z.y)
    nz = Vec(x.z,y.z,z.z)
    return Vec(nx,ny,nz)
end

@inline function adjoint(T::AbstractTen)
    x = T.x
    y = T.y
    z = T.z
    nx = Vec(conj(x.x), conj(y.x), conj(z.x))
    ny = Vec(conj(x.y), conj(y.y), conj(z.y))
    nz = Vec(conj(x.z), conj(y.z), conj(z.z))
    return Vec(nx,ny,nz)
end

@inline dot(v::AbstractVec{<:Any,1},T::AbstractTen) = dot(transpose(T),v)

@inline dotadd(u::AbstractVec,v::AbstractVec,a::Number) = @muladd u.x*v.x + u.y*v.y + u.z*v.z + a

@inline function muladd(T::AbstractTen,v::AbstractVec{<:Any,1},u::AbstractVec{<:Any,1})
    Tt = transpose(T)
    return Vec(dotadd(Tt.x,v,u.x), dotadd(Tt.y,v,u.y), dotadd(Tt.z,v,u.z))
end
@inline dotadd(T::AbstractTen,v::AbstractVec{<:Any,1},u::AbstractVec{<:Any,1}) = muladd(T,v,u)

@inline muladd(v::AbstractVec{<:Any,1},T::AbstractTen,u::AbstractVec{<:Any,1}) = Vec(dotadd(T.x,v,u.x), dotadd(T.y,v,u.y), dotadd(T.z,v,u.z))
@inline dotadd(v::AbstractVec{<:Any,1},T::AbstractTen,u::AbstractVec{<:Any,1}) = muladd(v,T,u)

@inline *(T::AbstractTen,B::AbstractTen) = Vec(T*B.x,T*B.y,T*B.z)

@inline muladd(A::AbstractTen, B::AbstractTen, C::AbstractTen) = Vec(muladd(A,B.x,C.x), muladd(A,B.y,C.y), muladd(A,B.z,C.z))
@inline dotadd(A::AbstractTen, B::AbstractTen, C::AbstractTen) = muladd(A,B,C)

@inline otimes(u::AbstractVec,v::AbstractVec) = Ten(xx=u.x*v.x, xy=u.x*v.y, xz=u.x*v.z,
                                                    yx=u.y*v.x, yy=u.y*v.y, yz=u.y*v.z,
                                                    zx=u.z*v.x, zy=u.z*v.y, zz=u.z*v.z)

const ⊗ = otimes