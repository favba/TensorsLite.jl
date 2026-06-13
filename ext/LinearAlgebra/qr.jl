struct QR{T, TQ, TR} <: LinearAlgebra.Factorization{T}
    Q::TQ
    R::TR
    @inline QR(Q::TQ, R::TR) where {TQ<:Ten, TR<:Ten} = new{nonzero_eltype(TQ), TQ, TR}(Q, R) 
end

Base.iterate(S::QR) = (S.Q, Val(:R))
Base.iterate(S::QR, ::Val{:R}) = (S.R, Val(:done))
Base.iterate(::QR, ::Val{:done}) = nothing

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, F::QR)
    summary(io, F); println(io)
    println(io, "Q rotation matrix:")
    show(io, mime, F.Q)
    println(io, "\nR matrix:")
    show(io, mime, F.R)
end

@inline LinearAlgebra.dot(q::QR, T::AbstractTensor) = LinearAlgebra.dot(q.Q, LinearAlgebra.dot(q.R,T))

@inline LinearAlgebra.dot(T::AbstractTensor, q::QR) = LinearAlgebra.dot(LinearAlgebra.dot(T, q.Q), q.R)

@inline Base.:*(q::QR, T::Union{<:Ten,<:Vec}) = q.Q * (q.R*T)

@inline Base.:*(T::Ten, q::QR) = (T*q.Q) * q.R

@inline Base.:\(q::QR, v::Vec) = q.R \ (q.Q' * v)

@inline _normalize(x) = copysign(one(x), x)

@inline _normalize(x::Complex) = x / abs(x)

@inline _conj_otimes(x) = otimes(x)
@inline _conj_otimes(x::Vec{<:Union{Zero,Complex}}) = otimes(x, conj(x))

@inline function housenholder(T::Ten, ::Val{P}) where {P}
    if P === :x
        a = T.x
        e1 = 𝐢
        a1 = a.x
    elseif P === :y
        a = Vec(Zero(), T.yy, T.zy)
        e1 = 𝐣
        a1 = a.y
    end

    if typeof(a) === Vec0D
        return 𝐈
    end

    σ = norm(a)
    α = -_normalize(a1)*σ
    v = a - α*e1
    iv = inner(v,v)
    β = ifelse(iszero(iv), zero(iv), 2 / iv)
    H = 𝐈 - β * _conj_otimes(v)
end

@inline function apply_housenholder(H::Ten, T::Ten, ::Val{P}) where {P}

    xx = muladd(H.xx, T.xx, muladd(H.xy, T.yx, H.xz*T.zx))
    xy = muladd(H.xx, T.xy, muladd(H.xy, T.yy, H.xz*T.zy))
    xz = muladd(H.xx, T.xz, muladd(H.xy, T.yz, H.xz*T.zz))
    yx = Zero()
    yy = muladd(H.yx, T.xy, muladd(H.yy, T.yy, H.yz*T.zy))
    yz = muladd(H.yx, T.xz, muladd(H.yy, T.yz, H.yz*T.zz))
    zx = Zero()
    if P === :y
        zy = Zero()
    else
        zy = muladd(H.zx, T.xy, muladd(H.zy, T.yy, H.zz*T.zy))
    end
    zz = muladd(H.zx, T.xz, muladd(H.zy, T.yz, H.zz*T.zz))

    return Ten(xx, xy, xz,
               yx, yy, yz,
               zx, zy, zz)
end

@inline function LinearAlgebra.qr(T::Ten)
    Vx = Val{:x}()
    Vy = Val{:y}()
    H1 = housenholder(T, Vx)
    T1 = apply_housenholder(H1, T, Vx)
    H2 = housenholder(T1, Vy)
    R = apply_housenholder(H2, T1, Vy)
    Q = H1*H2
    return QR(Q, R)
end

