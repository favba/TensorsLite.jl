
########################## aliases ###########################

const Vec{T} = AbstractTensor{1, T}

const Vec3D{T} = Tensor{1,T,T,T,T}
const Vec2Dxy{T} = Tensor{1, Union{Zero, T}, T, T, Zero}
const Vec2Dxz{T} = Tensor{1, Union{Zero, T}, T, Zero, T}
const Vec2Dyz{T} = Tensor{1, Union{Zero, T}, Zero, T, T}
const Vec2D{T} = Union{Vec2Dxy{T}, Vec2Dxz{T}, Vec2Dyz{T}}
const Vec1Dx{T} = Tensor{1, Union{Zero, T}, T, Zero, Zero}
const Vec1Dy{T} = Tensor{1, Union{Zero, T}, Zero, T, Zero}
const Vec1Dz{T} = Tensor{1, Union{Zero, T}, Zero, Zero, T}
const Vec1D{T} = Union{Vec1Dx{T}, Vec1Dy{T}, Vec1Dz{T}}
const VecND{T} = Union{Vec3D{T}, Vec2D{T}, Vec1D{T}}
const Vec0D = Vec3D{Zero}

# This ends up also being a VecMaybe1Dz{T, Tz}, if T===Zero and Tz != Zero
const VecMaybe2Dxy{T, Tz} = Tensor{1, Union{T, Tz}, T, T, Tz}

const Ten{T} = AbstractTensor{2, T}

const Ten3D{T} = Tensor{2,T,Vec3D{T},Vec3D{T},Vec3D{T}}
const Ten2Dxy{T} = Tensor{2, Union{Zero, T}, Vec2Dxy{T}, Vec2Dxy{T}, Vec0D}
const Ten2Dxz{T} = Tensor{2, Union{Zero, T}, Vec2Dxz{T}, Vec0D, Vec2Dxz{T}}
const Ten2Dyz{T} = Tensor{2, Union{Zero, T}, Vec0D, Vec2Dyz{T}, Vec2Dyz{T}}
const Ten2D{T} = Union{Ten2Dxy{T}, Ten2Dxz{T}, Ten2Dyz{T}}
const Ten1Dx{T} = Tensor{2, Union{Zero, T}, Vec1Dx{T}, Vec0D, Vec0D}
const Ten1Dy{T} = Tensor{2, Union{Zero, T}, Vec0D, Vec1Dy{T}, Vec0D}
const Ten1Dz{T} = Tensor{2, Union{Zero, T}, Vec0D, Vec0D, Vec1Dz{T}}
const Ten1D{T} = Union{Ten1Dx{T}, Ten1Dy{T}, Ten1Dz{T}}
const TenND{T} = Union{Ten3D{T}, Ten2D{T}, Ten1D{T}}
const Ten0D = Ten3D{Zero}
const DiagTen3D{T} = Tensor{2, Union{Zero, T}, Vec1Dx{T}, Vec1Dy{T}, Vec1Dz{T}}
const DiagTen2Dxy{T} = Tensor{2, Union{Zero, T}, Vec1Dx{T}, Vec1Dy{T}, Vec0D}
const DiagTen2Dxz{T} = Tensor{2, Union{Zero, T}, Vec1Dx{T}, Vec0D, Vec1Dz{T}}
const DiagTen2Dyz{T} = Tensor{2, Union{Zero, T}, Vec0D, Vec1Dy{T}, Vec1Dz{T}}
const DiagTen{T} = Union{DiagTen3D{T}, DiagTen2Dxy{T}, DiagTen2Dxz{T}, DiagTen2Dyz{T}}

const TenMaybe2Dxy{T, Tz} = Tensor{2, Union{T, Tz}, VecMaybe2Dxy{T, Tz}, VecMaybe2Dxy{T, Tz}, Vec3D{Tz}}

########################## aliases ###########################

########################## Vec constructors ###########################

@inline function Vec(a, b, c)
    if (a isa AbstractTensor || b isa AbstractTensor || c isa AbstractTensor)
        throw(ArgumentError("Tensors are not valid input to the `Vec` function"))
    end
    return Tensor(a, b, c)
end

Vec(;x=𝟎,y=𝟎,z=𝟎) = Vec(x,y,z)

Vec3D{T}(a, b, c) where {T} = Vec(convert(T, a), convert(T, b), convert(T, c))
Vec3D(a::T1, b::T2, c::T3) where {T1,T2,T3} = Vec3D{promote_type(T1,T2,T3)}(a, b, c)

Vec2Dxy{T}(a, b) where {T} = Vec(convert(T, a), convert(T, b), Zero())
Vec2Dxy(a, b) = Vec2Dxy{promote_type(typeof(a),typeof(b))}(a, b)

Vec2Dxz{T}(a, b) where {T} = Vec(convert(T, a), Zero(), convert(T, b))
Vec2Dxz(a, b) = Vec2Dxz{promote_type(typeof(a),typeof(b))}(a, b)

Vec2Dyz{T}(a, b) where {T} = Vec(Zero(), convert(T, a), convert(T, b))
Vec2Dyz(a, b) = Vec2Dyz{promote_type(typeof(a),typeof(b))}(a, b)

Vec1Dx{T}(a) where {T} = Vec(convert(T, a), Zero(), Zero())
Vec1Dx(a) = Vec1Dx{typeof(a)}(a)

Vec1Dy{T}(a) where {T} = Vec(Zero(), convert(T, a), Zero())
Vec1Dy(a) = Vec1Dy{typeof(a)}(a)

Vec1Dz{T}(a) where {T} = Vec(Zero(), Zero(), convert(T, a))
Vec1Dz(a) = Vec1Dz{typeof(a)}(a)

########################## Vec constructors ###########################

################ Ten constructors #####################

@inline Ten(x::Vec,y::Vec,z::Vec) = Tensor(x,y,z)

@inline function Ten(xx, xy, xz,
                     yx, yy, yz,
                     zx, zy, zz)
    x = Vec(xx, yx, zx)
    y = Vec(xy, yy, zy)
    z = Vec(xz, yz, zz)
    return Ten(x, y, z)
end

@inline Ten(; xx = 𝟎, yx = 𝟎, zx = 𝟎, xy = 𝟎, yy = 𝟎, zy = 𝟎, xz = 𝟎, yz = 𝟎, zz = 𝟎) = 
    Ten(xx, xy, xz,
        yx, yy, yz,
        zx, zy, zz)

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

################ Ten constructors #####################
