for TT in (DDiagTen, DDiagSymTen, TensorsLite.Ten0D, TensorsLite.SymTen0D)
    @inline LinearAlgebra.eigvals(T::TT) = Vec(T.xx, T.yy, T.zz)
    @inline LinearAlgebra.eigvecs(::TT) = 𝐈
    @inline LinearAlgebra.eigen(T::TT) = LinearAlgebra.Eigen(LinearAlgebra.eigvals(T), LinearAlgebra.eigvecs(T))
end

@inline _to_complex(x) = x < 0

@inline _to_complex(x::AbstractFloat) = signbit(x)

@inline _to_complex(x::Complex) = true

@inline function _eigvals_Ten2D(T11, T12, T21, T22)

    To2 = (T11 + T22)/2
    mD =  T12*T21 - T11*T22

    delta2 = muladd(To2, To2, mD)

    if _to_complex(delta2)
        deltac = fsqrt(complex(delta2))
        return (To2 - deltac, To2 + deltac)
    else
        delta = fsqrt(delta2)
        return (To2 - delta, To2 + delta)
    end

end

@inline LinearAlgebra.eigvals(T::QuasiTen2Dxy) =
    Vec2Dxy(_eigvals_Ten2D(T.xx, T.xy, T.yx, T.yy)...) + T.zz*𝐤

@inline LinearAlgebra.eigvals(T::QuasiTen2Dxz) =
    Vec2Dxz(_eigvals_Ten2D(T.xx, T.xz, T.zx, T.zz)...) + T.yy*𝐣

@inline LinearAlgebra.eigvals(T::QuasiTen2Dyz) =
    Vec2Dyz(_eigvals_Ten2D(T.yy, T.yz, T.zy, T.zz)...) + T.xx*𝐢

@inline function _eigvals_Sym2D(T11, T12, T22)
    To2 = (T11 + T22)/2
    mD =  T12*T12 - T11*T22

    delta = fsqrt(muladd(To2, To2, mD))

    return (To2 - delta, To2 + delta)
end

@inline LinearAlgebra.eigvals(S::QuasiSymTen2Dxy) =
    Vec2Dxy(_eigvals_Sym2D(S.xx, S.xy, S.yy)...) + S.zz*𝐤

@inline LinearAlgebra.eigvals(S::QuasiSymTen2Dxz) =
    Vec2Dxz(_eigvals_Sym2D(S.xx, S.xz, S.zz)...) + S.yy*𝐣

@inline LinearAlgebra.eigvals(S::QuasiSymTen2Dyz) =
    Vec2Dyz(_eigvals_Sym2D(S.yy, S.yz, S.zz)...) + S.xx*𝐢

@inline LinearAlgebra.eigvals(S::AntiSymTen2Dxy) = Vec2Dxy(-S.xy*im, S.xy*im)

@inline LinearAlgebra.eigvals(S::AntiSymTen2Dxz) = Vec2Dxz(-S.xz*im, S.xz*im)

@inline LinearAlgebra.eigvals(S::AntiSymTen2Dyz) = Vec2Dyz(-S.yz*im, S.yz*im)

@inline function LinearAlgebra.eigvals(W::AntiSymTen)
    w = norm(Vec(W.xy, W.xz, W.yz))
    return Vec2Dxz(-w*im, w*im)
end

@inline function LinearAlgebra.eigen(W::AntiSymTen)
    w = Vec(-W.yz, W.xz, -W.xy)
    nw = norm(w)
    l1 = -nw*im
    v1 = _eigvec(W, l1)
    return LinearAlgebra.Eigen(Vec2Dxz(l1, -l1), Ten(v1, w / nw, conj(v1)))
end

@inline function _eigen_Ten2D(T11, T12, T21, T22)

    To2 = (T11 + T22)/2
    mD =  T12*T21 - T11*T22

    delta2 = muladd(To2, To2, mD)

    if _to_complex(delta2)

        deltac = fsqrt(complex(delta2))
        L1c = To2 + deltac
        L2c = To2 - deltac
        Vc = if (T21 != zero(T21))
            (
                normalize((L1c - T22)*𝐢 + T21*𝐣),
                normalize((L2c - T22)*𝐢 + T21*𝐣)
            )
        else
            (
                normalize(T12*𝐢 + (L1c - T11)*𝐣),
                normalize(T12*𝐢 + (L2c - T11)*𝐣)
            )
        end

        return ((L2c, L1c), ((-Vc[2].x, -Vc[2].y), (Vc[1].x, Vc[1].y)))

    else

        delta = fsqrt(delta2)
        L1 = To2 + delta
        L2 = To2 - delta
        V = if (T21 != zero(T21))
            (
                normalize((L1 - T22)*𝐢 + T21*𝐣),
                normalize((L2 - T22)*𝐢 + T21*𝐣)
            )
        else
            (
                normalize(T12*𝐢 + (L1 - T11)*𝐣),
                normalize(T12*𝐢 + (L2 - T11)*𝐣)
            )
        end

        return ((L2, L1), ((-V[2].x, -V[2].y), (V[1].x, V[1].y)))

    end

end

function _dummy_ten_2d(T11, T12, T21, T22)
    To2 = (T11 + T22)/2
    mD =  T12*T21 - T11*T22
    delta2 = muladd(To2, To2, mD)
    delta = fsqrt(abs(delta2))
    return delta
end

@inline _out_eltype(T11, T12, T21, T22) = Base.promote_op(_dummy_ten_2d, typeof(T11), typeof(T12), typeof(T21), typeof(T22))

@inline @inline _out_eltype2(::Any, ::Type{To1}) where {To1} = To1
@inline @inline _out_eltype2(::Zero, ::Type{To1}) where {To1} = Zero
@inline @inline _out_eltype2(::One, ::Type{To1}) where {To1} = One

@inline function LinearAlgebra.eigen(T::QuasiTen2Dxy)
    evals, evecs = _eigen_Ten2D(T.xx, T.xy, T.yx, T.yy)
    e = Vec2Dxy(evals...) + T.zz*𝐤
    ev = Tensor(Vec2Dxy(evecs[1]...), Vec2Dxy(evecs[2]...), 𝐤)
    To1 = _out_eltype(T.xx, T.xy, T.yx, T.yy)
    CTo1 = Complex{To1}
    T2 = _out_eltype2(T.zz, To1)
    CTo2 = T2 === Zero ? Zero : CTo1
    out_type = Union{LinearAlgebra.Eigen{Union{One,Zero,To1}, Union{To1,T2}, Tensor{2, Union{One,Zero,To1}, Vec2Dxy{To1}, Vec2Dxy{To1}, Vec1Dz{One}} , Tensor{1,Union{To1,T2},To1,To1,T2}},
                 LinearAlgebra.Eigen{Union{One,Zero,CTo1}, Union{CTo1,CTo2}, Tensor{2, Union{One,Zero,CTo1}, Vec2Dxy{CTo1}, Vec2Dxy{CTo1}, Vec1Dz{One}} , Tensor{1,Union{CTo1,CTo2},CTo1,CTo1,CTo2}}}
    return LinearAlgebra.Eigen(e, ev)::out_type
end

@inline function LinearAlgebra.eigen(T::QuasiTen2Dxz)
    evals, evecs = _eigen_Ten2D(T.xx, T.xz, T.zx, T.zz)
    e = Vec2Dxz(evals...) + T.yy*𝐣
    ev = Tensor(Vec2Dxz(evecs[1]...), 𝐣, Vec2Dxz(evecs[2]...))
    To1 = _out_eltype(T.xx, T.xz, T.zx, T.zz)
    CTo1 = Complex{To1}
    T2 = _out_eltype2(T.yy, To1)
    CTo2 = T2 === Zero ? Zero : CTo1
    out_type = Union{LinearAlgebra.Eigen{Union{One,Zero,To1}, Union{To1,T2}, Tensor{2, Union{One,Zero,To1}, Vec2Dxz{To1}, Vec1Dy{One}, Vec2Dxz{To1}} , Tensor{1,Union{To1,T2},To1,T2,To1}},
                 LinearAlgebra.Eigen{Union{One,Zero,CTo1}, Union{CTo1,CTo2}, Tensor{2, Union{One,Zero,CTo1}, Vec2Dxz{CTo1}, Vec1Dy{One}, Vec2Dxz{CTo1}} , Tensor{1,Union{CTo1,CTo2},CTo1,CTo2,CTo1}}}
    return LinearAlgebra.Eigen(e, ev)::out_type
end

@inline function LinearAlgebra.eigen(T::QuasiTen2Dyz)
    evals, evecs = _eigen_Ten2D(T.yy, T.yz, T.zy, T.zz)
    e = Vec2Dyz(evals...) + T.xx*𝐢
    ev = Tensor(𝐢, Vec2Dyz(evecs[1]...), Vec2Dyz(evecs[2]...))
    To1 = _out_eltype(T.yy, T.yz, T.zy, T.zz)
    CTo1 = Complex{To1}
    T2 = _out_eltype2(T.xx, To1)
    CTo2 = T2 === Zero ? Zero : CTo1
    out_type = Union{LinearAlgebra.Eigen{Union{One,Zero,To1}, Union{To1,T2}, Tensor{2, Union{One,Zero,To1}, Vec1Dx{One}, Vec2Dyz{To1}, Vec2Dyz{To1}} , Tensor{1,Union{To1,T2},T2,To1,To1}},
                 LinearAlgebra.Eigen{Union{One,Zero,CTo1}, Union{CTo1,CTo2}, Tensor{2, Union{One,Zero,CTo1}, Vec1Dx{One}, Vec2Dyz{CTo1}, Vec2Dyz{CTo1}} , Tensor{1,Union{CTo1,CTo2},CTo2,CTo1,CTo1}}}
    return LinearAlgebra.Eigen(e, ev)::out_type
end

@inline function _eigen_SymTen2D(T11, T12, T22)

    To2 = (T11 + T22)/2
    mD =  T12*T12 - T11*T22

    delta = fsqrt(muladd(To2, To2, mD))
    L1 = To2 + delta
    L2 = To2 - delta

    u1 = normalize((L1 - T22)*𝐢 + T12*𝐣)

    u2 = 𝐤 × u1
    V = ((-u2.x, -u2.y), (u1.x, u1.y))

    return ((L2, L1), V)
end

@inline function LinearAlgebra.eigen(S::QuasiSymTen2Dxy)
    evals, evecs = _eigen_SymTen2D(S.xx, S.xy, S.yy)
    e = Vec2Dxy(evals...) + S.zz*𝐤
    ev = Tensor(Vec2Dxy(evecs[1]...), Vec2Dxy(evecs[2]...), 𝐤)
    return LinearAlgebra.Eigen(e, ev)
end

@inline function LinearAlgebra.eigen(S::QuasiSymTen2Dxz)
    evals, evecs = _eigen_SymTen2D(S.xx, S.xz, S.zz)
    e = Vec2Dxz(evals...) + S.yy*𝐣
    ev = Tensor(Vec2Dxz(evecs[1]...), 𝐣, Vec2Dxz(evecs[2]...))
    return LinearAlgebra.Eigen(e, ev)
end

@inline function LinearAlgebra.eigen(S::QuasiSymTen2Dyz)
    evals, evecs = _eigen_SymTen2D(S.yy, S.yz, S.zz)
    e = Vec2Dyz(evals...) + S.xx*𝐢
    ev = Tensor(𝐢, Vec2Dyz(evecs[1]...), Vec2Dyz(evecs[2]...))
    return LinearAlgebra.Eigen(e, ev)
end

@inline function LinearAlgebra.eigen(S::AntiSymTen2Dxy)
    S12 = S.xy
    mS12 = -S12
    L1 = S12*im
    L2 = mS12*im
    aux = oftype(S12, 0.707106781186547524400844362104849039284835937688474)
    maux = -aux
    v1 = Vec2Dxy(aux*im, maux)
    v2 = Vec2Dxy(maux*im, maux)
    return LinearAlgebra.Eigen(Vec2Dxy(L2, L1), Tensor(-v2,v1,𝐤))
end

@inline function LinearAlgebra.eigen(S::AntiSymTen2Dxz)
    S13 = S.xz
    mS13 = -S13
    L1 = S13*im
    L2 = mS13*im
    aux = oftype(S13, 0.707106781186547524400844362104849039284835937688474)
    maux = -aux
    v1 = Vec2Dxz(aux*im, maux)
    v2 = Vec2Dxz(maux*im, maux)
    return LinearAlgebra.Eigen(Vec2Dxz(L2, L1), Tensor(-v2, 𝐣, v1))
end

@inline function LinearAlgebra.eigen(S::AntiSymTen2Dyz)
    S23 = S.yz
    mS23 = -S23
    L1 = S23*im
    L2 = mS23*im
    aux = oftype(S23, 0.707106781186547524400844362104849039284835937688474)
    maux = -aux
    v1 = Vec2Dyz(aux*im, maux)
    v2 = Vec2Dyz(maux*im, maux)
    return LinearAlgebra.Eigen(Vec2Dyz(L2, L1), Tensor(𝐢, -v2, v1))
end

@inline _fix(r::T) where {T<:AbstractFloat} = max(-one(T), min(r, one(T)))

@inline _fix(r) = r

@inline function jacobi_rotation_coefficients(S::SymTen, ::Val{P}) where {P}

    if P === :xy
        a21 = S.xy
        a11 = S.xx
        a22 = S.yy
    elseif P === :xz
        a21 = S.xz
        a11 = S.xx
        a22 = S.zz
    elseif P === :yz
        a21 = S.yz
        a11 = S.yy
        a22 = S.zz
    end

    if typeof(a21) === Zero
        return (One(), Zero())
    end

    T = promote_type(typeof(a11), typeof(a22), typeof(a21))

    _iszero = iszero(a21)

    τ = (a22 - a11) / (2a21)

    t = copysign(one(T), τ) / (abs(τ) + fsqrt(1 + τ*τ))

    c = inv(fsqrt(1 + t*t))

    s = t*c

    c = ifelse(_iszero, one(c), c)

    s = ifelse(_iszero, zero(s), s)

    return (c, s)
end

@inline function build_jacobi_rotation_matrix(S::SymTen, vp::Val{P}) where {P}
    c, s = jacobi_rotation_coefficients(S, vp)
    if P === :xy
        return Ten(xx = c, xy = s, yx = -s, yy = c, zz = One())
    elseif P === :xz
        return Ten(xx = c, xz = s, zx = -s, zz = c, yy = One())
    elseif P === :yz
        return Ten(yy = c, yz = s, zy = -s, zz = c, xx = One())
    end
end

@inline function apply_jocobi_rotation(S::SymTen, R::Ten, ::Val{P}) where {P}
    SR = S*R
    Rt = transpose(R)

    xx = muladd(Rt.xx, SR.xx, muladd(Rt.xy, SR.yx, Rt.xz*SR.zx))
    yy = muladd(Rt.yx, SR.xy, muladd(Rt.yy, SR.yy, Rt.yz*SR.zy))
    zz = muladd(Rt.zx, SR.xz, muladd(Rt.zy, SR.yz, Rt.zz*SR.zz))

    if P === :xy
        xy = Zero()
    else
        xy = muladd(Rt.xx, SR.xy, muladd(Rt.xy, SR.yy, Rt.xz*SR.zy))
    end

    if P === :xz
        xz = Zero()
    else
        xz = muladd(Rt.xx, SR.xz, muladd(Rt.xy, SR.yz, Rt.xz*SR.zz))
    end

    if P === :yz
        yz = Zero()
    else
        yz = muladd(Rt.yx, SR.xz, muladd(Rt.yy, SR.yz, Rt.yz*SR.zz))
    end

    return SymTen(xx, xy, xz, yy, yz, zz)
end

@inline apply_jocobi_rotation(S::SymTen, V::Val{P}) where {P} = apply_jocobi_rotation(S, build_jacobi_rotation_matrix(S, V), V)

@inline function sort_eigen(v, T)
    a = v.x
    b = v.y
    c = v.z

    at = T.x
    bt = T.y
    ct = T.z

    m1 = a > b

    a, b = ifelse(m1, b, a), ifelse(m1, a, b)
    at, bt = ifelse(m1, bt, at), ifelse(m1, at, bt)

    m2 = b > c

    b, c = ifelse(m2, c, b), ifelse(m2, b, c)
    bt, ct = ifelse(m2, ct, bt), ifelse(m2, bt, ct)

    m3 = a > b

    a, b = ifelse(m3, b, a), ifelse(m3, a, b)
    at, bt = ifelse(m3, bt, at), ifelse(m3, at, bt)

    return (Vec(a,b,c), Ten(at, bt, ct))
end

@inline _not_converged(S::SymTen{<:Number}, a_tol) = 2*(abs(S.xy) + abs(S.xz) + abs(S.yz)) >= a_tol

#For SIMD
@inline _not_converged(S::SymTen, a_tol) = all(2*(abs(S.xy) + abs(S.xz) + abs(S.yz)) >= a_tol)

@inline function LinearAlgebra.eigen(S::SymTen)
    Vxy = Val{:xy}()
    Vxz = Val{:xz}()
    Vyz = Val{:yz}()

    a_tol = eps(norm(S))

    _Gxy = build_jacobi_rotation_matrix(S, Vxy)
    _S1 = apply_jocobi_rotation(S, _Gxy, Vxy)
    _V1 = _Gxy

    _Gxz = build_jacobi_rotation_matrix(_S1, Vxz)
    _S2 = apply_jocobi_rotation(_S1, _Gxz, Vxz)
    _V2 = _V1 * _Gxz

    _Gyz = build_jacobi_rotation_matrix(_S2, Vyz)

    S3 = apply_jocobi_rotation(_S2, _Gyz, Vyz)
    V3 = _V2 * _Gyz

    i = 1
    while _not_converged(S3, a_tol) && (i <= 9)
        Gxy = build_jacobi_rotation_matrix(S3, Vxy)
        S1 = apply_jocobi_rotation(S3, Gxy, Vxy)
        V1 = V3 * Gxy

        Gxz = build_jacobi_rotation_matrix(S1, Vxz)
        S2 = apply_jocobi_rotation(S1, Gxz, Vxz)
        V2 = V1 * Gxz

        Gyz = build_jacobi_rotation_matrix(S2, Vyz)

        S3 = apply_jocobi_rotation(S2, Gyz, Vyz)
        V3 = V2 * Gyz
        i+=1
    end

    _λ = Vec(S3.xx, S3.yy, S3.zz)

    λ, Vf = sort_eigen(_λ, V3)

    return LinearAlgebra.Eigen(λ, Vf)
end

@inline function LinearAlgebra.eigvals(S::SymTen)
    Vxy = Val{:xy}()
    Vxz = Val{:xz}()
    Vyz = Val{:yz}()

    a_tol = eps(norm(S))

    _S1 = apply_jocobi_rotation(S, Vxy)

    _S2 = apply_jocobi_rotation(_S1, Vxz)

    S3 = apply_jocobi_rotation(_S2, Vyz)

    i = 1
    while _not_converged(S3, a_tol) && (i <= 9)
        S1 = apply_jocobi_rotation(S3, Vxy)

        S2 = apply_jocobi_rotation(S1, Vxz)

        S3 = apply_jocobi_rotation(S2, Vyz)
        i+=1
    end

    λ = sort(Vec(S3.xx, S3.yy, S3.zz))

    return λ
end

@inline LinearAlgebra.eigvecs(S::Ten) = LinearAlgebra.eigen(S).vectors

@inline function givens_zx(T::Ten)
    x = T.yx
    y = T.zx

    r = hypot(x, y)

    mask = iszero(r)

    c = ifelse(mask, one(r), x/r)
    s = ifelse(mask, zero(r), -y/r)

    return Ten(xx=One(), yy=c, yz=s, zy=-s, zz=c)
end

@inline function apply_givens_zx(G, T)
    TG = T*G
    Gt = G'
    y = Gt*TG.y
    z = Gt*TG.z
    _aux = Ten(xx=Gt.xx, xy=Gt.xy, xz=Gt.xz,
               yx=Gt.yx, yy=Gt.yy, yz=Gt.yz)
    x = _aux*TG.x
    return Ten(x, y, z)
end

@inline zero_zx(T) = apply_givens_zx(givens_zx(T), T)

@inline function _eigvals(A::Ten)

    a_tol = eps(norm(A))

    H = zero_zx(A)

    T = Ten(yy=H.yy, yz=H.yz, zy=H.zy, zz=H.zz)

    s = LinearAlgebra.tr(T)
    t = LinearAlgebra.det(T + 𝐢𝐢)

    x = muladd(H, H.x, -s*H.x)  + t*𝐢

    P = housenholder(Ten(x, Vec0D(), Vec0D()), Val{:x}())

    Hb = P'*H*P
    H1 = zero_zx(Hb)

    i=1
    while !((abs(H1.yx) <= a_tol) || (abs(H1.zy) <= a_tol) ) && (i <= 20)

        T1 = Ten(yy=H1.yy, yz=H1.yz, zy=H1.zy, zz=H1.zz)

        s1 = LinearAlgebra.tr(T1)
        t1 = LinearAlgebra.det(T1 + 𝐢𝐢)

        x1 = muladd(H1, H1.x, -s1*H1.x)  + t1*𝐢

        P1 = housenholder(Ten(x1, Vec0D(), Vec0D()), Val{:x}())

        Hb1 = P1'*H1*P1
        H1 = zero_zx(Hb1)

        i += 1
    end

    if ((abs(H1.yx) <= a_tol) && (abs(H1.zy) <= a_tol))

        return sort(Vec(H1.xx, H1.yy, H1.zz))

    elseif (abs(H1.yx) < abs(H1.zy))

        λ2d1 = LinearAlgebra.eigvals(Ten(xx=H1.yy, xy=H1.yz, yx=H1.zy, yy=H1.zz))
        return Vec(H1.xx, λ2d1.x, λ2d1.y)

    else

        λ2d2 = LinearAlgebra.eigvals(Ten(xx=H1.xx, xy=H1.xy, yx=H1.yx, yy=H1.yy))
        return Vec(H1.zz, λ2d2.x, λ2d2.y)

    end
end

@inline LinearAlgebra.eigvals(A::Ten) = _eigvals(A)

@inline function pick_highest_norm(v1, v2, v3)
    nv1 = norm(v1)
    nv2 = norm(v2)
    nv3 = norm(v3)

    if nv1 > nv2
        if nv1 > nv3
            return v1
        else
            return v3
        end
    else
        if nv2 > nv3
            return v2
        else
            return v3
        end
    end
end

@inline function _eigvec(T::Ten, λ)
    M = T - λ*𝐈
    Mt = transpose(M)
    return normalize(pick_highest_norm(Mt.x × Mt.y, Mt.y × Mt.z, Mt.z × Mt.x))
end

@inline function _eigen(Te::Ten)
    λ = LinearAlgebra.eigvals(Te)

    #Degenerate case
    if λ.x ≈ λ.y ≈ λ.z
        return LinearAlgebra.Eigen(λ, convert(Ten3D{nonzero_eltype(λ)}, 𝐈))
    end

    v1 = _eigvec(Te, λ.x)
    v2 = _eigvec(Te, λ.y)
    v3 = _eigvec(Te, λ.z)

    ev = Ten(v1, v2, v3)

    return LinearAlgebra.Eigen(λ, ev)
end

#Help inference
@inline LinearAlgebra.eigen(R::Ten{T}) where {T <: Real} = _eigen(R)::Union{LinearAlgebra.Eigen{T, T, Ten3D{T}, Vec3D{T}}, LinearAlgebra.Eigen{Complex{T}, Complex{T}, Ten3D{Complex{T}}, Vec3D{Complex{T}}}}

@inline function _eigvals_LA(A::Ten{Complex{T}}) where {T}
    v = LinearAlgebra.eigvals(Array(A))
    return Vec(v[1], v[2], v[3])::Union{Vec3D{T}, Vec3D{Complex{T}}}
end

@inline function _eigen_LA(A::Ten{Complex{T}}) where {T}
    v, vecs = LinearAlgebra.eigen(Array(A))
    @inbounds LinearAlgebra.Eigen(Vec(v[1], v[2], v[3])::Union{Vec3D{T}, Vec3D{Complex{T}}},
                                  Ten(vecs[1,1], vecs[1,2], vecs[1,3],
                                      vecs[2,1], vecs[2,2], vecs[2,3],
                                      vecs[3,1], vecs[3,2], vecs[3,3])
    )
end

#ToDo - implement optimized algorithm
@inline LinearAlgebra.eigvals(S::Ten{<:Complex}) = _eigvals_LA(S)
@inline LinearAlgebra.eigvals(S::SymTen{<:Complex}) = _eigvals_LA(S)
@inline LinearAlgebra.eigen(S::Ten{<:Complex}) = _eigen_LA(S)
@inline LinearAlgebra.eigen(S::SymTen{<:Complex}) = _eigen_LA(S)
