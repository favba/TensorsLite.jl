for TT in (DDiagTen, DDiagSymTen, TensorsLite.Ten0D, TensorsLite.SymTen0D)
    @inline LinearAlgebra.eigvals(T::TT) = Vec(T.xx, T.yy, T.zz)
    @inline LinearAlgebra.eigvecs(::TT) = 𝐈
    @inline LinearAlgebra.eigen(T::TT) = LinearAlgebra.Eigen(LinearAlgebra.eigvals(T), LinearAlgebra.eigvecs(T))
end

@inline _isnegative(x) = x < 0

@inline _isnegative(x::AbstractFloat) = signbit(x)

@inline function _eigvals_Ten2D(T11, T12, T21, T22)

    To2 = (T11 + T22)/2
    mD =  T12*T21 - T11*T22 

    delta2 = muladd(To2, To2, mD)

    if _isnegative(delta2)
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


@inline function _eigen_Ten2D(T11, T12, T21, T22)

    To2 = (T11 + T22)/2
    mD =  T12*T21 - T11*T22 

    delta2 = muladd(To2, To2, mD)

    if _isnegative(delta2)

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

@inline _out_eltype(::Type{T}) where {T} = Base.promote_op(fsqrt, T)

@inline function LinearAlgebra.eigen(T::QuasiTen2Dxy)
    evals, evecs = _eigen_Ten2D(T.xx, T.xy, T.yx, T.yy)
    e = Vec2Dxy(evals...) + T.zz*𝐤
    ev = Tensor(Vec2Dxy(evecs[1]...), Vec2Dxy(evecs[2]...), 𝐤)
    To1 = _out_eltype(promote_type(typeof(T.xx), typeof(T.xy), typeof(T.yx), typeof(T.yy)))
    CTo1 = Complex{To1}
    T2 = typeof(T.zz)
    CTo2 = T2 === Zero ? Zero : CTo1
    out_type = Union{LinearAlgebra.Eigen{Union{One,Zero,To1}, Union{To1,T2}, Tensor{2, Union{One,Zero,To1}, Vec2Dxy{To1}, Vec2Dxy{To1}, Vec1Dz{One}} , Tensor{1,Union{To1,T2},To1,To1,T2}},
                 LinearAlgebra.Eigen{Union{One,Zero,CTo1}, Union{CTo1,CTo2}, Tensor{2, Union{One,Zero,CTo1}, Vec2Dxy{CTo1}, Vec2Dxy{CTo1}, Vec1Dz{One}} , Tensor{1,Union{CTo1,CTo2},CTo1,CTo1,CTo2}}}
    return LinearAlgebra.Eigen(e, ev)::out_type
end

@inline function LinearAlgebra.eigen(T::QuasiTen2Dxz)
    evals, evecs = _eigen_Ten2D(T.xx, T.xz, T.zx, T.zz)
    e = Vec2Dxz(evals...) + T.yy*𝐣
    ev = Tensor(Vec2Dxz(evecs[1]...), 𝐣, Vec2Dxz(evecs[2]...))
    To1 = _out_eltype(promote_type(typeof(T.xx), typeof(T.xz), typeof(T.zx), typeof(T.zz)))
    CTo1 = Complex{To1}
    T2 = typeof(T.yy)
    CTo2 = T2 === Zero ? Zero : CTo1
    out_type = Union{LinearAlgebra.Eigen{Union{One,Zero,To1}, Union{To1,T2}, Tensor{2, Union{One,Zero,To1}, Vec2Dxz{To1}, Vec1Dy{One}, Vec2Dxz{To1}} , Tensor{1,Union{To1,T2},To1,T2,To1}},
                 LinearAlgebra.Eigen{Union{One,Zero,CTo1}, Union{CTo1,CTo2}, Tensor{2, Union{One,Zero,CTo1}, Vec2Dxz{CTo1}, Vec1Dy{One}, Vec2Dxz{CTo1}} , Tensor{1,Union{CTo1,CTo2},CTo1,CTo2,CTo1}}}
    return LinearAlgebra.Eigen(e, ev)::out_type
end

@inline function LinearAlgebra.eigen(T::QuasiTen2Dyz)
    evals, evecs = _eigen_Ten2D(T.yy, T.yz, T.zy, T.zz)
    e = Vec2Dyz(evals...) + T.xx*𝐢
    ev = Tensor(𝐢, Vec2Dyz(evecs[1]...), Vec2Dyz(evecs[2]...))
    To1 = _out_eltype(promote_type(typeof(T.yy), typeof(T.yz), typeof(T.zy), typeof(T.zz)))
    CTo1 = Complex{To1}
    T2 = typeof(T.xx)
    CTo2 = T2 === Zero ? Zero : CTo1
    out_type = Union{LinearAlgebra.Eigen{Union{One,Zero,To1}, Union{To1,T2}, Tensor{2, Union{One,Zero,To1}, Vec1Dx{One}, Vec2Dyz{To1}, Vec2Dyz{To1}} , Tensor{1,Union{To1,T2},T2,To1,To1}},
                 LinearAlgebra.Eigen{Union{One,Zero,CTo1}, Union{CTo1,CTo2}, Tensor{2, Union{One,Zero,CTo1}, Vec1Dx{One}, Vec2Dyz{CTo1}, Vec2Dyz{CTo1}} , Tensor{1,Union{CTo1,CTo2},CTo2,CTo1,CTo1}}}
    return LinearAlgebra.Eigen(e, ev)::out_type
end

@inline function _eigen_SymTen2D(T11, T12, T22)

    To2 = (T11 + T22)/2
    mD =  T12*T12 - T11*T22 

    delta2 = muladd(To2, To2, mD)
    delta = fsqrt(delta2)
    L1 = To2 + delta
    L2 = To2 - delta

    u1 = normalize((L1 - T22)*𝐢 + T12*𝐣)

    u2 = cross(𝐤, u1)
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

@inline function LinearAlgebra.eigvals(S::SymTen{<:Number})
    e11 = S.xx
    e12 = S.xy
    e13 = S.xz
    e22 = S.yy
    e23 = S.yz
    e33 = S.zz

    q = (e11 + e22 + e33) / 3

    e12p2 = e12^2
    e13p2 = e13^2
    e23p2 = e23^2

    e11mq = (e11-q)
    e22mq = (e22-q)
    e33mq = (e33-q)

    p1 = e12p2 + e13p2 + e23p2
    p2 = muladd(e11mq, e11mq, muladd(e22mq, e22mq, muladd(e33mq, e33mq, 2*p1)))
    p = fsqrt(p2 / 6)


    #r = ((e11mq)*(e22mq)*(e33mq) - (e11mq)*(e23p2) - (e12p2)*(e33mq) + 2*(e12*e13*e23) - (e13p2)*(e22mq))/(2*p*p*p)
    r = ( muladd(e11mq, e23p2, muladd(e22mq, e13p2, e33mq*e12p2)) - muladd(e11mq, e22mq*e33mq, 2*(e12*e13*e23)) ) / ((-2)*p*p*p)
  
    # In exact arithmetic for a symmetric matrix  -1 <= r <= 1
    # but computation error can leave it slightly outside this range.
    r = _fix(r)

    ϕ = acos(r) / 3

    aux = 2*p
  
    # the eigenvalues satisfy eig.z >= eig.y >= eig.x
    #eig3 = q + 2*p*cos(ϕ)
    eig3 = muladd(aux, cos(ϕ), q)
    # cos(x+y) = cos(x)*cos(y) - sin(x)*sin(y)
    #eig1 = q + 2*p*cos(ϕ+(2*π/3))  # q - 2*p*(cos(ϕ)/2 + (√3/2)sin(ϕ))
    eig1 = muladd(aux, cos(ϕ+(2*oftype(ϕ,π)/3)), q)  # q - 2*p*(cos(ϕ)/2 + (√3/2)sin(ϕ))
    # eig2 = 3*q - eig1 - eig3     # since trace(E) = eig.x + eig.y + eig.z = 3q
    eig2 = - muladd(-3,q, eig1 + eig3)     # since trace(E) = eig.x + eig.y + eig.z = 3q

    return Vec(eig1,eig2,eig3)
end

@inline function jacobi_rotation_coefficients(S::SymTen, ::Val{P}) where {P}

    if P === :xy
        a12 = S.xy
        a11 = S.xx
        a22 = S.yy
    elseif P === :xz
        a12 = S.xz
        a11 = S.xx
        a22 = S.zz
    elseif P === :yz
        a12 = S.yz
        a11 = S.yy
        a22 = S.zz
    end

    if typeof(a12) === Zero
        T = promote_type(typeof(a11), typeof(a22))
        return (One(), Zero())
    end

    T = promote_type(typeof(a11), typeof(a22), typeof(a12))

    _iszero = iszero(a12)

    τ = (a22 - a11) / (2a12)

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

    a_tol = eps(maximum(abs, S))

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
    while _not_converged(S3, a_tol) && (i <= 10)
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

    a_tol = eps(maximum(abs, S))

    _Gxy = build_jacobi_rotation_matrix(S, Vxy)
    _S1 = apply_jocobi_rotation(S, _Gxy, Vxy)

    _Gxz = build_jacobi_rotation_matrix(_S1, Vxz) 
    _S2 = apply_jocobi_rotation(_S1, _Gxz, Vxz) 

    _Gyz = build_jacobi_rotation_matrix(_S2, Vyz) 

    S3 = apply_jocobi_rotation(_S2, _Gyz, Vyz)

    while _not_converged(S3, a_tol) && (i <= 9)
        Gxy = build_jacobi_rotation_matrix(S3, Vxy)
        S1 = apply_jocobi_rotation(S3, Gxy, Vxy)

        Gxz = build_jacobi_rotation_matrix(S1, Vxz) 
        S2 = apply_jocobi_rotation(S1, Gxz, Vxz) 

        Gyz = build_jacobi_rotation_matrix(S2, Vyz) 

        S3 = apply_jocobi_rotation(S2, Gyz, Vyz)
        i+=1
    end

    λ = sort(Vec(S3.xx, S3.yy, S3.zz))

    return λ
end

@inline LinearAlgebra.eigvecs(S::Ten) = LinearAlgebra.eigen(S).vectors
