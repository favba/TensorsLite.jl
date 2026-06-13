struct LU{T, TL, TU} <: LinearAlgebra.Factorization{T}
    L::TL
    U::TU
    @inline LU(L::TL, U::TU) where {TL<:Ten, TU<:Ten} = new{nonzero_eltype(TU), TL, TU}(L, U)
end

Base.iterate(S::LU) = (S.L, Val(:U))
Base.iterate(S::LU, ::Val{:U}) = (S.U, Val(:done))
Base.iterate(::LU, ::Val{:done}) = nothing

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, F::LU)
    summary(io, F); println(io)
    println(io, "L factor:")
    show(io, mime, F.L)
    println(io, "\nU factor:")
    show(io, mime, F.U)
end

@inline LinearAlgebra.dot(f::LU, T::AbstractTensor) = LinearAlgebra.dot(f.L, LinearAlgebra.dot(f.U,T))

@inline LinearAlgebra.dot(T::AbstractTensor, f::LU) = LinearAlgebra.dot(LinearAlgebra.dot(T, f.L), f.U)

@inline Base.:*(f::LU, T::Union{<:Ten,<:Vec}) = f.L * (f.U*T)

@inline Base.:*(T::Ten, f::LU) = (T*f.L) * f.U

@inline Base.:\(F::LU, v::Vec) = F.U \ (F.L \ v)

_isZero(x) = false
_isZero(::Zero) = true

@inline function LinearAlgebra.lu(T::Ten)
    u11, u12, u13 = (T.xx, T.xy, T.xz)

    l21 = _isZero(u11) ? Zero() : T.yx / u11
    l31 = _isZero(u11) ? Zero() : T.zx / u11

    ml21 = -l21
    u22 = muladd(ml21, u12, T.yy)
    u23 = muladd(ml21, u13, T.yz)

    l32 = _isZero(u22) ? Zero() : (T.zy - l31*u12) / u22

    mTzz = -T.zz
    u33 = - muladd(l31, u13, muladd(l32, u23, mTzz))

    L = Ten(xx=One(),
            yx=l21, yy=One(),
            zx=l31, zy=l32, zz=One())

    U = Ten(xx=u11, xy=u12, xz=u13,
                    yy=u22, yz=u23,
                            zz=u33)

    return LU(L,U)
end
