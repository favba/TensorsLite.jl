module TensorsLiteLinearAlgebraExt

using TensorsLite, Zeros

import TensorsLite: dot, *, +, -, _muladd, Ten1D, SymTen1D, DiagTen, DiagSymTen

import LinearAlgebra: LinearAlgebra, norm, ⋅, cross, normalize

@inline LinearAlgebra.dot(a::AbstractTensor, b::AbstractTensor) = dot(a, b)
@inline LinearAlgebra.dot(x::Vec, A::Ten, y::Vec) = dot(dot(x, A), y)

@inline fsqrt(x) = @fastmath sqrt(x)

@inline LinearAlgebra.norm(u::AbstractTensor) = fsqrt(real(inner(u, u)))

@inline LinearAlgebra.norm(u::Vec1Dx, ::Real = 2) = abs(u.x)

@inline LinearAlgebra.norm(u::Vec1Dy, ::Real = 2) = abs(u.y)

@inline LinearAlgebra.norm(u::Vec1Dz, ::Real = 2) = abs(u.z)

@inline LinearAlgebra.norm(::VecND{Zero}, ::Real = 2) = 𝟎

@inline LinearAlgebra.normalize(u::AbstractTensor) = u / norm(u)

@inline LinearAlgebra.normalize(u::AbstractTensor{N,Zero}) where {N} = u

@inline function LinearAlgebra.cross(a::Vec, b::Vec)
    ax = a.x
    ay = a.y
    az = a.z
    bx = b.x
    by = b.y
    bz = b.z
    return  Vec(_muladd(ay, bz, -(az * by)), _muladd(az, bx, -(ax * bz)), _muladd(ax, by, -(ay * bx)))
end

base_type(::Type{T}) where {T} = T
base_type(::Type{Complex{T}}) where {T} = T
@inline Base.isapprox(x::AbstractTensor{N}, y::AbstractTensor{N}) where {N} = norm(x - y) <= max(Base.rtoldefault(base_type(nonzero_eltype(x))), Base.rtoldefault(base_type(nonzero_eltype(y)))) * max(norm(x), norm(y))

@inline LinearAlgebra.tr(T::Ten) = T.xx + T.yy + T.zz

@inline function LinearAlgebra.det(T::Ten)

    Txx = T.xx
    Tyx = T.yx
    Tzx = T.zx

    Txy = T.xy
    Tyy = T.yy
    Tzy = T.zy

    Txz = T.xz
    Tyz = T.yz
    Tzz = T.zz

    return _muladd((Txy*Tyz - Txz*Tyy), Tzx, _muladd((Tzy*Txz - Tzz*Txy), Tyx, (Tzz*Tyy - Tzy*Tyz )*Txx))

end

@inline LinearAlgebra.det(::AntiSymTen) = 𝟎

@inline LinearAlgebra.eigvals(T::Union{<:Ten1D,<:SymTen1D,<:DiagTen,<:DiagSymTen}) = Vec(T.xx, T.yy, T.zz)

@inline LinearAlgebra.eigvecs(::Union{<:Ten1D,<:SymTen1D,<:DiagTen,<:DiagSymTen}) = 𝐈

@inline LinearAlgebra.eigen(T::Union{<:Ten1D,<:SymTen1D,<:DiagTen,<:DiagSymTen}) = LinearAlgebra.Eigen(LinearAlgebra.eigvals(T), LinearAlgebra.eigvecs(T))

@inline _isnegative(x) = x < 0

@inline _isnegative(x::AbstractFloat) = signbit(x)

@inline function _eigvals_Ten2D(T11, T12, T21, T22)

    To2 = (T11 + T22)/2
    mD =  T12*T21 - T11*T22 

    delta2 = _muladd(To2, To2, mD)

    if _isnegative(delta2)
        deltac = fsqrt(complex(delta2))
        return (To2 + deltac, To2 - deltac)
    else
        delta = fsqrt(delta2)
        return (To2 + delta, To2 - delta)
    end

end

@inline LinearAlgebra.eigvals(T::Tensor{2, Union{Zero, T1, T2}, Vec2Dxy{T1}, Vec2Dxy{T1}, Vec1Dz{T2}}) where {T1, T2} =
    Vec2Dxy(_eigvals_Ten2D(T.xx, T.xy, T.yx, T.yy)...) + T.zz*𝐤

@inline LinearAlgebra.eigvals(T::Tensor{2, Union{Zero, T1, T2}, Vec2Dxz{T1}, Vec1Dy{T2}, Vec2Dxz{T1}}) where {T1, T2} =
    Vec2Dxz(_eigvals_Ten2D(T.xx, T.xz, T.zx, T.zz)...) + T.yy*𝐣

@inline LinearAlgebra.eigvals(T::Tensor{2, Union{Zero, T1, T2}, Vec1Dx{T2}, Vec2Dyz{T1}, Vec2Dyz{T1}}) where {T1, T2} =
    Vec2Dyz(_eigvals_Ten2D(T.yy, T.yz, T.zy, T.zz)...) + T.xx*𝐢

@inline function _eigvals_Sym2D(T11, T12, T22)
    To2 = (T11 + T22)/2
    mD =  T12*T12 - T11*T22 

    delta = fsqrt(_muladd(To2, To2, mD))

    return (To2 + delta, To2 - delta)
end
                                                                        #xx, xy, xz, yy, yz, zz
@inline LinearAlgebra.eigvals(S::SymmetricTensor{2, Union{Zero, T1, T2}, T1, T1, Zero, T1, Zero, T2 }) where {T1, T2} =
    Vec2Dxy(_eigvals_Sym2D(S.xx, S.xy, S.yy)...) + S.zz*𝐤

@inline LinearAlgebra.eigvals(S::SymmetricTensor{2, Union{Zero, T1, T2}, T1, Zero, T1, T2, Zero, T1 }) where {T1, T2} =
    Vec2Dxz(_eigvals_Sym2D(S.xx, S.xz, S.zz)...) + S.yy*𝐣

@inline LinearAlgebra.eigvals(S::SymmetricTensor{2, Union{Zero, T1, T2}, T2, Zero, Zero, T1, T1, T1 }) where {T1, T2} =
    Vec2Dyz(_eigvals_Sym2D(S.yy, S.yz, S.zz)...) + S.xx*𝐢

@inline LinearAlgebra.eigvals(S::AntiSymTen2Dxy) = Vec2Dxy(S.xy*im, -S.xy*im)

@inline LinearAlgebra.eigvals(S::AntiSymTen2Dxz) = Vec2Dxz(S.xz*im, -S.xz*im)

@inline LinearAlgebra.eigvals(S::AntiSymTen2Dyz) = Vec2Dyz(S.yz*im, -S.yz*im)


@inline function _eigen_Ten2D(T11, T12, T21, T22)

    To2 = (T11 + T22)/2
    mD =  T12*T21 - T11*T22 

    delta2 = _muladd(To2, To2, mD)

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

        return ((L1c, L2c), ((Vc[1].x, Vc[1].y), (Vc[2].x, Vc[2].y)))

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

        return ((L1, L2), ((V[1].x, V[1].y), (V[2].x, V[2].y)))

    end

end

@inline function _out_eltype(::Type{T}) where {T<:Number}
    return Base.promote_op(fsqrt, T)
end

@inline function LinearAlgebra.eigen(T::Tensor{2, Union{Zero, T1, T2}, Vec2Dxy{T1}, Vec2Dxy{T1}, Vec1Dz{T2}}) where {T1, T2}
    evals, evecs = _eigen_Ten2D(T.xx, T.xy, T.yx, T.yy)
    e = Vec2Dxy(evals...) + T.zz*𝐤
    ev = Tensor(Vec2Dxy(evecs[1]...), Vec2Dxy(evecs[2]...), 𝐤)
    To1 = _out_eltype(T1)
    CTo1 = Complex{To1}
    CTo2 = T2 === Zero ? Zero : CTo1
    out_type = Union{LinearAlgebra.Eigen{Union{One,Zero,To1}, Union{To1,T2}, Tensor{2, Union{One,Zero,To1}, Vec2Dxy{To1}, Vec2Dxy{To1}, Vec1Dz{One}} , Tensor{1,Union{To1,T2},To1,To1,T2}},
                 LinearAlgebra.Eigen{Union{One,Zero,CTo1}, Union{CTo1,CTo2}, Tensor{2, Union{One,Zero,CTo1}, Vec2Dxy{CTo1}, Vec2Dxy{CTo1}, Vec1Dz{One}} , Tensor{1,Union{CTo1,CTo2},CTo1,CTo1,CTo2}}}
    return LinearAlgebra.Eigen(e, ev)::out_type
end

@inline function LinearAlgebra.eigen(T::Tensor{2, Union{Zero, T1, T2}, Vec2Dxz{T1}, Vec1Dy{T2}, Vec2Dxz{T1}}) where {T1, T2}
    evals, evecs = _eigen_Ten2D(T.xx, T.xz, T.zx, T.zz)
    e = Vec2Dxz(evals...) + T.yy*𝐣
    ev = Tensor(Vec2Dxz(evecs[1]...), 𝐣, Vec2Dxz(evecs[2]...))
    To1 = _out_eltype(T1)
    CTo1 = Complex{To1}
    CTo2 = T2 === Zero ? Zero : CTo1
    out_type = Union{LinearAlgebra.Eigen{Union{One,Zero,To1}, Union{To1,T2}, Tensor{2, Union{One,Zero,To1}, Vec2Dxz{To1}, Vec1Dy{One}, Vec2Dxz{To1}} , Tensor{1,Union{To1,T2},To1,T2,To1}},
                 LinearAlgebra.Eigen{Union{One,Zero,CTo1}, Union{CTo1,CTo2}, Tensor{2, Union{One,Zero,CTo1}, Vec2Dxz{CTo1}, Vec1Dy{One}, Vec2Dxz{CTo1}} , Tensor{1,Union{CTo1,CTo2},CTo1,CTo2,CTo1}}}
    return LinearAlgebra.Eigen(e, ev)::out_type
end

@inline function LinearAlgebra.eigen(T::Tensor{2, Union{Zero, T1, T2}, Vec1Dx{T2}, Vec2Dyz{T1}, Vec2Dyz{T1}}) where {T1, T2}
    evals, evecs = _eigen_Ten2D(T.yy, T.yz, T.zy, T.zz)
    e = Vec2Dyz(evals...) + T.xx*𝐢
    ev = Tensor(𝐢, Vec2Dyz(evecs[1]...), Vec2Dyz(evecs[2]...))
    To1 = _out_eltype(T1)
    CTo1 = Complex{To1}
    CTo2 = T2 === Zero ? Zero : CTo1
    out_type = Union{LinearAlgebra.Eigen{Union{One,Zero,To1}, Union{To1,T2}, Tensor{2, Union{One,Zero,To1}, Vec1Dx{One}, Vec2Dyz{To1}, Vec2Dyz{To1}} , Tensor{1,Union{To1,T2},T2,To1,To1}},
                 LinearAlgebra.Eigen{Union{One,Zero,CTo1}, Union{CTo1,CTo2}, Tensor{2, Union{One,Zero,CTo1}, Vec1Dx{One}, Vec2Dyz{CTo1}, Vec2Dyz{CTo1}} , Tensor{1,Union{CTo1,CTo2},CTo2,CTo1,CTo1}}}
    return LinearAlgebra.Eigen(e, ev)::out_type
end

@inline function _eigen_SymTen2D(T11, T12, T22)

    To2 = (T11 + T22)/2
    mD =  T12*T12 - T11*T22 

    delta2 = _muladd(To2, To2, mD)
    delta = fsqrt(delta2)
    L1 = To2 + delta
    L2 = To2 - delta

    u1 = normalize((L1 - T22)*𝐢 + T12*𝐣)

    u2 = cross(𝐤, u1)
    V = ((u1.x, u1.y), (u2.x, u2.y))

    return ((L1, L2), V)
end

@inline function LinearAlgebra.eigen(S::SymmetricTensor{2, Union{Zero, T1, T2}, T1, T1, Zero, T1, Zero, T2 }) where {T1, T2}
    evals, evecs = _eigen_SymTen2D(S.xx, S.xy, S.yy)
    e = Vec2Dxy(evals...) + S.zz*𝐤
    ev = Tensor(Vec2Dxy(evecs[1]...), Vec2Dxy(evecs[2]...), 𝐤)
    return LinearAlgebra.Eigen(e, ev)
end

@inline function LinearAlgebra.eigen(S::SymmetricTensor{2, Union{Zero, T1, T2}, T1, Zero, T1, T2, Zero, T1 }) where {T1, T2}
    evals, evecs = _eigen_SymTen2D(S.xx, S.xz, S.zz)
    e = Vec2Dxz(evals...) + S.yy*𝐣
    ev = Tensor(Vec2Dxz(evecs[1]...), 𝐣, Vec2Dxz(evecs[2]...))
    return LinearAlgebra.Eigen(e, ev)
end

@inline function LinearAlgebra.eigen(S::SymmetricTensor{2, Union{Zero, T1, T2}, T2, Zero, Zero, T1, T1, T1 }) where {T1, T2}
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
    return LinearAlgebra.Eigen(Vec2Dxy(L1, L2), Tensor(v1,v2,𝐤))
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
    return LinearAlgebra.Eigen(Vec2Dxz(L1, L2), Tensor(v1, 𝐣, v2))
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
    return LinearAlgebra.Eigen(Vec2Dyz(L1, L2), Tensor(𝐢, v1, v2))
end

@inline _fix(r::T) where {T<:AbstractFloat} = max(-one(T), min(r, one(T)))

@inline _fix(r) = r

@inline function LinearAlgebra.eigvals(S::SymTen)
    e11 = S.xx
    e12 = S.xy
    e13 = S.xz
    e22 = S.yy
    e23 = S.yz
    e33 = S.zz

    q = inv(3)*(e11 + e22 + e33)

    e12p2 = e12^2
    e13p2 = e13^2
    e23p2 = e23^2

    e11mq = (e11-q)
    e22mq = (e22-q)
    e33mq = (e33-q)

    p1 = e12p2 + e13p2 + e23p2
    p2 = _muladd(e11mq, e11mq, _muladd(e22mq, e22mq, _muladd(e33mq, e33mq, 2*p1)))
    p = fsqrt(inv(6)*p2)


    #r = ((e11mq)*(e22mq)*(e33mq) - (e11mq)*(e23p2) - (e12p2)*(e33mq) + 2*(e12*e13*e23) - (e13p2)*(e22mq))/(2*p*p*p)
    r = ( _muladd(e11mq, e23p2, _muladd(e22mq, e13p2, e33mq*e12p2)) - _muladd(e11mq, e22mq*e33mq, 2*(e12*e13*e23)) ) / ((-2)*p*p*p)
  
    # In exact arithmetic for a symmetric matrix  -1 <= r <= 1
    # but computation error can leave it slightly outside this range.
    r = _fix(r)

    ϕ = inv(3)*acos(r)

    aux = 2*p
  
    # the eigenvalues satisfy eig.z >= eig.y >= eig.x
    #eig3 = q + 2*p*cos(ϕ)
    eig3 = _muladd(aux, cos(ϕ), q)
    # cos(x+y) = cos(x)*cos(y) - sin(x)*sin(y)
    #eig1 = q + 2*p*cos(ϕ+(2*π/3))  # q - 2*p*(cos(ϕ)/2 + (√3/2)sin(ϕ))
    eig1 = _muladd(aux, cos(ϕ+(2*π/3)), q)  # q - 2*p*(cos(ϕ)/2 + (√3/2)sin(ϕ))
    # eig2 = 3*q - eig1 - eig3     # since trace(E) = eig.x + eig.y + eig.z = 3q
    eig2 = - _muladd(-3,q, eig1 + eig3)     # since trace(E) = eig.x + eig.y + eig.z = 3q

    return Vec(eig1,eig2,eig3)
end

@inline function LinearAlgebra.eigen(S::SymTen)

    λ1, λ2, λ3 = LinearAlgebra.eigvals(S)

    ######################### This part was adapted from https://github.com/KristofferC/Tensors.jl/blob/master/src/eigen.jl #################################

      # Calculate the first eigenvector
        # This should be orthogonal to these three rows of A - λ1*I
        # Use all combinations of cross products and choose the "best" one
    R = S - λ1*𝐈

    r1 = R.x
    r2 = R.y
    r3 = R.z

    n1 = r1 ⋅ r1
    n2 = r2 ⋅ r2
    n3 = r3 ⋅ r3

    r12 = cross(r1, r2)
    r23 = cross(r2, r3)
    r31 = cross(r3, r1)

    n12 = r12 ⋅ r12
    n23 = r23 ⋅ r23
    n31 = r31 ⋅ r31

    # we want best angle so we put all norms on same footing
    # (cheaper to multiply by third nᵢ rather than divide by the two involved)

    nr12 = r12 / fsqrt(n12)
    nr31 = r31 / fsqrt(n31)
    nr23 = r23 / fsqrt(n23)

    ϕ1 = ifelse(n12 * n3 > n23 * n1,
        ifelse(n12 * n3 > n31 * n2,
            nr12,
            nr31
        ),
        ifelse(n23 * n1 > n31 * n2,
            nr23,
            nr31
        )
    )

    # Calculate the second eigenvector
    # This should be orthogonal to the previous eigenvector and the three
    # rows of A - λ2*I. However, we need to "solve" the remaining 2x2 subspace
    # problem in case the cross products are identically or nearly zero

    # The remaing 2x2 subspace is:
    abs2x = abs2(ϕ1.x)
    abs2y = abs2(ϕ1.y)
    abs2z = abs2(ϕ1.z)

    aux_flag = abs2x < abs2y

    r1 = inv(fsqrt(abs2x + abs2z))
    r2 = inv(fsqrt(abs2y + abs2z))

    ze = zero(ϕ1.x)
    ox = ifelse(aux_flag, -ϕ1.z*r1, ze)
    oy = ifelse(aux_flag, ze, ϕ1.z*r2)
    oz = ifelse(aux_flag, ϕ1.x*r1, -ϕ1.y*r2)
    
    orthogonal1 = Vec(ox,oy,oz)

    orthogonal2 = cross(ϕ1, orthogonal1)

    # The projected 2x2 eigenvalue problem is C x = 0 where C is the projection
    # of (A - λ2*I) onto the subspace {orthogonal1, orthogonal2}
    a_orth_1 = S ⋅ orthogonal1

    a_orth_2 = S ⋅ orthogonal2

    mλ2 = -λ2
    c11 = dotadd(orthogonal1, a_orth_1, mλ2)
    c12 = orthogonal1 ⋅ a_orth_2
    c22 = dotadd(orthogonal2, a_orth_2, mλ2)

    # Solve this robustly (some values might be small or zero)
    c11² = abs2(c11)
    c12² = abs2(c12)
    c22² = abs2(c22)

    # Boolean flags
    b11_22 = c11² >= c22²
    b11_12 = c11² >= c12²
    b22_12 = c22² >= c12²
    b_nonzero = (c11² > 0) | (c12² > 0)

    # Select numerator/denominator
    num = ifelse(b11_22,
        ifelse(b11_12,
            c12,
            c11
        ),
        ifelse(b22_12,
            c12,
            c22
        )
    )

    den = ifelse(b11_22,
        ifelse(b11_12,
            c11,
            c12
        ),
        ifelse(b22_12,
            c22,
            c12
        )
    )

    tmp = num / den
    invnorm = inv(fsqrt(1 + abs2(tmp)))
    tmp_invnorm = tmp * invnorm

    p1 = ifelse(b11_22,
        ifelse(b11_12,
            tmp_invnorm,
            invnorm
        ),
        ifelse(b22_12,
            invnorm,
            tmp_invnorm
        )
    )

    p2 = ifelse(b11_22,
        ifelse(b11_12,
            invnorm,
            tmp_invnorm
        ),
        ifelse(b22_12,
            tmp_invnorm,
            invnorm
        )
    )

    ϕ2 = ifelse(b11_22 & !b_nonzero,
        orthogonal1,
        p1*orthogonal1 - p2*orthogonal2
    )
    
    # The third eigenvector is a simple cross product of the other two
    ϕ3 = cross(ϕ1, ϕ2) # should be normalized already
     
    return LinearAlgebra.Eigen(Vec3D(λ1,λ2,λ3),Tensor(ϕ1,ϕ2,ϕ3))
end

end
