import LinearAlgebra: transpose, adjoint

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

@inline function muladd(T::AbstractTen,v::AbstractVec{<:Any,1},u::AbstractVec{<:Any,1})
    @muladd begin
        x = T.x.x*v.x + T.y.x*v.y + T.z.x*v.z + u.x
        y = T.x.y*v.x + T.y.y*v.y + T.z.y*v.z + u.y
        z = T.x.z*v.x + T.y.z*v.y + T.z.z*v.z + u.z
    end
    return Vec(x,y,z)
end

@inline muladd(v::AbstractVec{<:Any,1},T::AbstractTen,u::AbstractVec{<:Any,1}) = muladd(transpose(T),v,u)

@inline *(T::AbstractTen,B::AbstractTen) = Vec(T*B.x,T*B.y,T*B.z)

@inline muladd(A::AbstractTen, B::AbstractTen, C::AbstractTen) = Vec(muladd(A,B.x,C.x), muladd(A,B.y,C.y), muladd(A,B.z,C.z))