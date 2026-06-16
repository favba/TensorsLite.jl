
struct SVD{T, TU, TS, TV} <: LinearAlgebra.Factorization{T}
    U::TU
    S::TS
    V::TV
    @inline SVD(U::TU, S::Vec{T}, V::TV) where {T, TU<:Ten, TV<:Ten} = new{T, TU, typeof(S), TV}(U, S, V)
end

@inline function Base.getproperty(F::SVD, s::Symbol)
    if s === :Vt
        return transpose(getfield(F, :V))
    else
        return getfield(F, s)
    end
end

Base.iterate(S::SVD) = (S.U, Val(:S))
Base.iterate(S::SVD, ::Val{:S}) = (S.S, Val(:Vt))
Base.iterate(S::SVD, ::Val{:Vt}) = (S.Vt, Val(:done))
Base.iterate(S::SVD, ::Val{:done}) = nothing

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, F::SVD{<:Any,<:Ten,<:Vec,<:Ten})
    summary(io, F); println(io)
    println(io, "U factor:")
    show(io, mime, F.U)
    println(io, "\nsingular values:")
    show(io, mime, F.S)
    println(io, "\nV factor:")
    show(io, mime, F.V)
end

_try_pinv(::Zero) = Zero()

_try_pinv(x) = ifelse(iszero(x), x, inv(x))

_try_pinv(::Zero, tol) = Zero()

_try_pinv(x, tol) = ifelse(abs(x) > tol, inv(x), zero(x))

@inline _max_to_min(v::Vec, T::Ten) = (Vec(v.z, v.y, v.x), Ten(-T.z, T.y, T.x))
@inline _max_to_min(v::DVec2Dxy, T::Ten) = (Vec(v.y, v.x, v.z), Ten(-T.y, T.x, T.z))
@inline _max_to_min(v::DVec2Dxz, T::Ten) = (Vec(v.z, v.x, v.y), Ten(-T.z, T.x, T.y))
@inline _max_to_min(v::DVec2Dyz, T::Ten) = (Vec(v.z, v.y, v.x), Ten(-T.z, T.y, T.x))
@inline _max_to_min(v::DVec1Dx, T::Ten) = (v, T)
@inline _max_to_min(v::DVec1Dy, T::Ten) = (Vec(v.y, v.z, v.x), Ten(-T.y, T.z, T.x))
@inline _max_to_min(v::DVec1Dz, T::Ten) = (Vec(v.z, v.y, v.x), Ten(-T.z, T.y, T.x))
@inline _max_to_min(v::Vec0D, T::Ten) = (v, T)

@inline _to_zero(x, tol) = ifelse(abs(x) > tol, x, zero(x))
@inline _to_zero(x::Union{Zero, One}, tol) = x

@inline _fix_rank(U::Ten, ::Ten) = U
#Ten2Dxy
@inline _fix_rank(U::Ten, ::Tensor{2, <:Any, <:DVec2Dxy, <:DVec2Dxy, Vec1Dz{One}}) = Ten(U.x, U.y, 𝐤)
#Ten2Dxz
@inline _fix_rank(U::Ten, ::Tensor{2, <:Any, <:DVec2Dxz, <:DVec2Dxz, Vec1Dy{One}}) = Ten(U.x, U.y, 𝐣)
#Ten2Dyz
@inline _fix_rank(U::Ten, ::Tensor{2, <:Any, <:DVec2Dyz, <:DVec2Dyz, Vec1Dx{One}}) = Ten(U.x, U.y, 𝐢)

@inline _eps(λ::Vec) = eps(nonzero_eltype(λ)(λ.x))
@inline _eps(λ::Vec0D) = Zero()

@inline _convert_ones(v::Vec) = Vec(1.0*v.x, 1.0*v.y, 1.0*v.z)

@inline function _svd(T::Ten)

    _eig, vecs = LinearAlgebra.eigen(tdot(T))
    eig = _convert_ones(_eig)

    λ, V = _max_to_min(eig, vecs)

    tol = _eps(λ)
    λ = Vec(_to_zero(λ.x, tol), _to_zero(λ.y, tol), _to_zero(λ.z, tol))

    σ = map(fsqrt, λ)

    Σinv = TensorsLite.DiagTen(_try_pinv(σ.x), _try_pinv(σ.y), _try_pinv(σ.z))

    _U = dot(T, dot(V, Σinv))

    U = _fix_rank(_U, V)

    return SVD(U, σ, V)
end

@inline LinearAlgebra.svd(T::Ten) = _svd(T)

@inline function _svdvals(T::Ten)
    _eig = LinearAlgebra.eigvals(tdot(T))
    eig = _convert_ones(_eig)

    λ = sort(eig) |> reverse

    tol = _eps(λ)
    λ = Vec(_to_zero(λ.x, tol), _to_zero(λ.y, tol), _to_zero(λ.z, tol))

    return map(fsqrt, λ)
end

@inline LinearAlgebra.svdvals(T::Ten) = _svdvals(T)

@inline function LinearAlgebra.rank(A::Ten;
                                    atol = 0.0,
                                    rtol = (eps(real(float(oneunit(nonzero_eltype(A)))))*3*iszero(atol)))

    s = LinearAlgebra.svdvals(A)
    tol = max(atol, rtol*s.x)
    r = 0
    r += ifelse(s.x > tol, 1, 0)
    r += ifelse(s.y > tol, 1, 0)
    r += ifelse(s.z > tol, 1, 0)

    return r
end
