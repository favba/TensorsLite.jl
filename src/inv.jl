@inline function cross(a::Vec, b::Vec)
    ax = a.x
    ay = a.y
    az = a.z
    bx = b.x
    by = b.y
    bz = b.z
    return  Vec(muladd(ay, bz, -(az * by)), muladd(az, bx, -(ax * bz)), muladd(ax, by, -(ay * bx)))
end

_try_inv(::Zero) = Zero()
_try_inv(x) = inv(x)

@inline _inv(T::Ten0D) = T
 
@inline _inv(T::DDiagTen) = Ten(Vec1Dx(_try_inv(T.xx)), Vec1Dy(_try_inv(T.yy)), Vec1Dz(_try_inv(T.zz))) 

@inline function _inv2D(a, b, c, d)
    mb = -b
    det = muladd(mb, c, a*d)
    idet = inv(det)

    return (d*idet, mb*idet, -c*idet,  a*idet)
end

@inline function _inv(A::QuasiTen2Dxy)
    iTxx, iTxy, iTyx, iTyy = _inv2D(A.xx, A.xy, A.yx, A.yy)
    return Ten(Vec2Dxy(iTxx, iTyx), Vec2Dxy(iTxy, iTyy), Vec1Dz(_try_inv(A.zz)))
end

@inline function _inv(A::QuasiTen2Dxz)
    iTxx, iTxz, iTzx, iTzz = _inv2D(A.xx, A.xz, A.zx, A.zz)
    return Ten(Vec2Dxz(iTxx, iTzx), Vec1Dy(_try_inv(A.yy)), Vec2Dxz(iTxz, iTzz))
end

@inline function _inv(A::QuasiTen2Dyz)
    iTyy, iTyz, iTzy, iTzz = _inv2D(A.yy, A.yz, A.zy, A.zz)
    return Ten(Vec1Dx(_try_inv(A.xx)), Vec2Dyz(iTyy, iTzy), Vec2Dyz(iTyz, iTzz))
end

@inline function _inv(T::Ten)

    x0 = T.x
    x1 = T.y
    x2 = T.z

    y0 = cross(x1,x2)
    d  = inv(dot(x0,y0))
    x0 = d * x0
    y0 = d * y0
    y1 = cross(x2,x0)
    y2 = cross(x0,x1)

    return transpose(Ten(y0, y1, y2))
end

@inline _inv(S::SymTen0D) = S

@inline _inv(S::DDiagSymTen) = SymTen(_try_inv(S.xx), Zero(), Zero(), _try_inv(S.yy), Zero(), _try_inv(S.zz))

@inline function _inv2D(a, b, d)
    mb = -b
    det = muladd(mb,b, a*d)
    idet = inv(det)

    return (d*idet, mb*idet,  a*idet)
end

@inline function _inv(S::QuasiSymTen2Dxy)
    iSxx, iSxy, iSyy = _inv2D(S.xx, S.xy, S.yy)
    return SymTen(iSxx, iSxy, Zero(), iSyy, Zero(), _try_inv(S.zz))
end

@inline function _inv(S::QuasiSymTen2Dxz)
    iSxx, iSxz, iSzz = _inv2D(S.xx, S.xz, S.zz)
    return SymTen(iSxx, Zero(), iSxz, _try_inv(S.yy), Zero(), iSzz)
end

@inline function _inv(S::QuasiSymTen2Dyz)
    iSyy, iSyz, iSzz = _inv2D(S.yy, S.yz, S.zz)
    return SymTen(_try_inv(S.xx), Zero(), Zero(), iSyy, iSyz, iSzz)
end

@inline function _inv(S::SymTen)
    a, b, c, d, e, f = TensorsLite.sym_ten_fields(S)

    mb = -b
    mc = -c
    me = -e

    m11 = muladd(d, f, me*e)
    m12 = muladd(c, e, mb*f)
    m13 = muladd(b, e, mc*d)

    m22 = muladd(a, f, mc*c)
    m23 = muladd(b, c, me*a)

    m33 = muladd(a, d, mb*b)

    det = muladd(a, m11, muladd(b, m12, c*m13))

    idet = inv(det)

    return idet * SymTen(m11, m12, m13, m22, m23, m33)
end

@inline Base.inv(T::Ten) = _inv(T)

