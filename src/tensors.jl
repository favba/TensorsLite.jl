
################ Ten constructors #####################
#
@inline function Ten(xx, xy, xz,
                     yx, yy, yz,
                     zx, zy, zz)
    x = Tensor(xx, yx, zx)
    y = Tensor(xy, yy, zy)
    z = Tensor(xz, yz, zz)
    return Tensor(x, y, z)
end

@inline Ten(; xx = 𝟎, yx = 𝟎, zx = 𝟎, xy = 𝟎, yy = 𝟎, zy = 𝟎, xz = 𝟎, yz = 𝟎, zz = 𝟎) = 
    Ten(xx, xy, xz,
        yx, yy, yz,
        zx, zy, zz)

Ten{T}(xx, xy, xz, yx, yy, yz, zx, zy, zz) where {T} = 
    Ten(convert(T, xx), convert(T, xy), convert(T, xz),
        convert(T, yx), convert(T, yy), convert(T, yz),
        convert(T, zx), convert(T, zy), convert(T, zz))

Ten3D{T}(xx, xy, xz, yx, yy, yz, zx, zy, zz) where {T} = 
    Ten(convert(T, xx), convert(T, xy), convert(T, xz),
        convert(T, yx), convert(T, yy), convert(T, yz),
        convert(T, zx), convert(T, zy), convert(T, zz))

Ten3D(xx, xy, xz, yx, yy, yz, zx, zy, zz) = 
    Ten3D{promote_type(typeof(xx), typeof(xy), typeof(xz),
                       typeof(yx), typeof(yy), typeof(yz),
                       typeof(zx), typeof(zy), typeof(zz))}(xx, xy, xz,
                                                            yx, yy, yz,
                                                            zx, zy, zz)

Ten2Dxy{T}(xx, xy, yx, yy) where {T} = 
    Ten(convert(T, xx), convert(T, xy), 𝟎,
        convert(T, yx), convert(T, yy), 𝟎,
        𝟎,              𝟎,              𝟎)

Ten2Dxy(xx, xy, yx, yy) = 
    Ten2Dxy{promote_type(typeof(xx), typeof(xy),
                         typeof(yx), typeof(yy))}(xx, xy,
                                                  yx, yy)


Ten2Dxz{T}(xx, xz, zx, zz) where {T} = 
    Ten(convert(T, xx), 𝟎, convert(T, xz),
        𝟎,              𝟎, 𝟎,
        convert(T, zx), 𝟎, convert(T, zz))

Ten2Dxz(xx, xz, zx, zz) = 
    Ten2Dxz{promote_type(typeof(xx), typeof(xz),
                         typeof(zx), typeof(zz))}(xx, xz,
                                                  zx, zz)


Ten2Dyz{T}(yy, yz, zy, zz) where {T} = 
    Ten(𝟎, 𝟎,              𝟎,
        𝟎, convert(T, yy), convert(T, yz),
        𝟎, convert(T, zy), convert(T, zz))

Ten2Dyz(yy, yz, zy, zz) = 
Ten2Dyz{promote_type(typeof(yy), typeof(yz),
                     typeof(zy), typeof(zz))}(yy, yz,
                                              zy, zz)


Ten1Dx{T}(xx) where {T} = Ten(convert(T, xx), 𝟎, 𝟎,
                              𝟎,              𝟎, 𝟎,
                              𝟎,              𝟎, 𝟎)

Ten1Dx(xx) = Ten1Dx{typeof(xx)}(xx)


Ten1Dy{T}(yy) where {T} = Ten(𝟎, 𝟎,              𝟎,
                              𝟎, convert(T, yy), 𝟎,
                              𝟎, 𝟎,              𝟎)

Ten1Dy(yy) = Ten1Dy{typeof(yy)}(yy)


Ten1Dz{T}(zz) where {T} = Ten(𝟎, 𝟎, 𝟎,
                              𝟎, 𝟎, 𝟎,
                              𝟎, 𝟎, convert(T, zz))

Ten1Dz(zz) = Ten1Dz{typeof(zz)}(zz)

#################### AbstractMatrix interface ############################

@inline Base.:*(T::Ten, v::AbstractTensor{<:Any, 1}) = dot(T, v)
@inline Base.:*(v::AbstractTensor{<:Any, 1}, T::Ten) = dot(v, T)

@inline function transpose(T::Ten)
    x = T.x
    y = T.y
    z = T.z
    nx = Tensor(x.x, y.x, z.x)
    ny = Tensor(x.y, y.y, z.y)
    nz = Tensor(x.z, y.z, z.z)
    return Tensor(nx, ny, nz)
end

@inline function adjoint(T::Ten)
    x = T.x
    y = T.y
    z = T.z
    nx = Tensor(conj(x.x), conj(y.x), conj(z.x))
    ny = Tensor(conj(x.y), conj(y.y), conj(z.y))
    nz = Tensor(conj(x.z), conj(y.z), conj(z.z))
    return Tensor(nx, ny, nz)
end

@inline dot(v::AbstractTensor{<:Any, 1}, T::Ten) = dot(transpose(T), v)

@inline function _muladd(T::Ten, v::AbstractTensor{<:Any, 1}, u::AbstractTensor{<:Any, 1})
    Tt = transpose(T)
    return Tensor(dotadd(Tt.x, v, u.x), dotadd(Tt.y, v, u.y), dotadd(Tt.z, v, u.z))
end
@inline muladd(T::Ten, v::AbstractTensor{<:Any, 1}, u::AbstractTensor{<:Any, 1}) = _muladd(T, v, u)
@inline dotadd(T::Ten, v::AbstractTensor{<:Any, 1}, u::AbstractTensor{<:Any, 1}) = _muladd(T, v, u)

@inline _muladd(v::AbstractTensor{<:Any, 1}, T::Ten, u::AbstractTensor{<:Any, 1}) = Tensor(dotadd(T.x, v, u.x), dotadd(T.y, v, u.y), dotadd(T.z, v, u.z))
@inline muladd(v::AbstractTensor{<:Any, 1}, T::Ten, u::AbstractTensor{<:Any, 1}) = _muladd(v, T, u)
@inline dotadd(v::AbstractTensor{<:Any, 1}, T::Ten, u::AbstractTensor{<:Any, 1}) = _muladd(v, T, u)

@inline Base.:*(T::Ten, B::Ten) = Tensor(T * B.x, T * B.y, T * B.z)
@inline dot(T::Ten, B::Ten) = T * B

@inline _muladd(A::Ten, B::Ten, C::Ten) = Tensor(_muladd(A, B.x, C.x), _muladd(A, B.y, C.y), _muladd(A, B.z, C.z))
@inline muladd(A::Ten, B::Ten, C::Ten) = _muladd(A, B, C)
@inline dotadd(A::Ten, B::Ten, C::Ten) = _muladd(A, B, C)

@inline otimes(u::AbstractTensor{<:Any, 1}, v::AbstractTensor{<:Any, 1}) = Ten(
    u.x * v.x, u.x * v.y, u.x * v.z,
    u.y * v.x, u.y * v.y, u.y * v.z,
    u.z * v.x, u.z * v.y, u.z * v.z
)

const ⊗ = otimes
