_try_inv(::Zero) = Zero()
_try_inv(x) = inv(x)

function _inv end

for TT in (DiagTen, Ten1Dx, Ten1Dy, Ten1Dz)
    @inline TensorsLiteLinearAlgebraExt._inv(T::TT) = Ten(Vec1Dx(_try_inv(T.xx)), Vec1Dy(_try_inv(T.yy)), Vec1Dz(_try_inv(T.zz))) 
end

@inline function _inv2D(a, b, c, d)
    mb = -b
    det = muladd(mb, c, a*d)
    idet = inv(det)

    return (d*idet, mb*idet, -c*idet,  a*idet)
end

@inline function _inv(T::Tensor{2, Union{T1,Tz,Zero}, Vec2Dxy{T1}, Vec2Dxy{T1}, Vec1Dz{Tz}}) where {T1, Tz}
    iTxx, iTxy, iTyx, iTyy = _inv2D(T.xx, T.xy, T.yx, T.yy)
    return Ten(Vec2Dxy(iTxx, iTyx), Vec2Dxy(iTxy, iTyy), Vec1Dz(_try_inv(T.zz)))
end

@inline function _inv(T::Tensor{2, Union{T1,Ty,Zero}, Vec2Dxz{T1}, Vec1Dy{Ty}, Vec2Dxz{T1}}) where {T1, Ty}
    iTxx, iTxz, iTzx, iTzz = _inv2D(T.xx, T.xz, T.zx, T.zz)
    return Ten(Vec2Dxz(iTxx, iTzx), Vec1Dy(_try_inv(T.yy)), Vec2Dxz(iTxz, iTzz))
end

@inline function _inv(T::Tensor{2, Union{T1,Tx,Zero}, Vec1Dx{Tx}, Vec2Dyz{T1}, Vec2Dyz{T1}}) where {T1, Tx}
    iTyy, iTyz, iTzy, iTzz = _inv2D(T.yy, T.yz, T.zy, T.zz)
    return Ten(Vec1Dx(_try_inv(T.xx)), Vec2Dyz(iTyy, iTzy), Vec2Dyz(iTyz, iTzz))
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

for TT in (DiagSymTen, SymTen1Dx, SymTen1Dy, SymTen1Dz)
    @inline TensorsLiteLinearAlgebraExt._inv(T::TT) = SymTen(_try_inv(T.xx), Zero(), Zero(), _try_inv(T.yy), Zero(), _try_inv(T.zz))
end

@inline function _inv2D(a, b, d)
    mb = -b
    det = muladd(mb,b, a*d)
    idet = inv(det)

    return (d*idet, mb*idet,  a*idet)
end

@inline function _inv(S::SymmetricTensor{2, Union{T,Tz,Zero}, T, T, Zero, T, Zero, Tz}) where {T, Tz}
    iSxx, iSxy, iSyy = _inv2D(S.xx, S.xy, S.yy)
    return SymTen(iSxx, iSxy, Zero(), iSyy, Zero(), _try_inv(S.zz))
end

@inline function _inv(S::SymmetricTensor{2, Union{T,Ty,Zero}, T, Zero, T, Ty, Zero, T}) where {T, Ty}
    iSxx, iSxz, iSzz = _inv2D(S.xx, S.xz, S.zz)
    return SymTen(iSxx, Zero(), iSxz, _try_inv(S.yy), Zero(), iSzz)
end

@inline function _inv(S::SymmetricTensor{2, Union{T,Tx,Zero}, Tx, Zero, Zero, T, T, T}) where {T, Tx}
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

@inline LinearAlgebra.inv(T::Ten) = _inv(T)

