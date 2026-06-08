@inline Base.reduce(op::F, v::Vec) where {F <: Function} = @inline op(op(v.x, v.y), v.z)

#Fix ambiguities
@inline Base.reduce(::typeof(hcat), v::Vec{<:Union{<:AbstractVector, <:AbstractMatrix}}) = @inline reduce(hcat, Array(v))
@inline Base.reduce(::typeof(vcat), v::Vec{<:Union{<:AbstractVector, <:AbstractMatrix}}) = @inline reduce(vcat, Array(v))

@inline Base.reduce(op::F, v::AbstractTensor) where {F <: Function} = @inline op(op(reduce(op, v.x), reduce(op, v.y)), reduce(op, v.z))

@inline Base.sum(v::Vec) = v.x + v.y + v.z

@inline Base.sum(v::AbstractTensor) = sum(v.x) + sum(v.y) + sum(v.z)

@inline Base.sum(op::F, v::Vec) where {F <: Function} = @inline op(v.x) + op(v.y) + op(v.z)

@inline Base.sum(op::F, v::AbstractTensor) where {F <: Function} = @inline sum(op,v.x) + sum(op,v.y) + sum(op,v.z)

@inline Base.map(f::F, vecs::Vararg{Vec}) where {F <: Function} = @inline(Tensor(f(_x.(vecs)...), f(_y.(vecs)...), f(_z.(vecs)...)))

@inline Base.map(f::F, vecs::Vararg{AbstractTensor{N}}) where {F <: Function, N} = @inline(Tensor(map(f, _x.(vecs)...), map(f, _y.(vecs)...), map(f, _z.(vecs)...)))

@inline Base.maximum(v::Vec) = max(v.x, v.y, v.z)

@inline Base.maximum(v::AbstractTensor) = max(maximum(v.x), maximum(v.y), maximum(v.z))

@inline Base.maximum(f::F, v::Vec) where {F <: Function} = max(f(v.x), f(v.y), f(v.z))

@inline Base.maximum(f::F, v::AbstractTensor) where {F <: Function} = max(maximum(f, v.x), maximum(f, v.y), maximum(f, v.z))

@inline Base.minimum(v::Vec) = min(v.x, v.y, v.z)

@inline Base.minimum(v::AbstractTensor) = min(minimum(v.x), minimum(v.y), minimum(v.z))

@inline Base.minimum(f::F, v::Vec) where {F <: Function} = min(f(v.x), f(v.y), f(v.z))

@inline Base.minimum(f::F, v::AbstractTensor) where {F <: Function} = min(minimum(f, v.x), minimum(f, v.y), minimum(f, v.z))

@inline Base.any(f::F, v::Vec) where {F <: Function} = f(v.x) || f(v.y) || f(v.z)

@inline Base.any(f::F, v::AbstractTensor) where {F <: Function} = any(f, v.x) || any(f, v.y) || any(f, v.z)

@inline Base.all(f::F, v::Vec) where {F <: Function} = f(v.x) && f(v.y) && f(v.z)

@inline Base.all(f::F, v::AbstractTensor) where {F <: Function} = all(f, v.x) && all(f, v.y) && all(f, v.z)

@inline _xx(u::AbstractTensor) = u.xx
@inline _xy(u::AbstractTensor) = u.xy
@inline _xz(u::AbstractTensor) = u.xz
@inline _yy(u::AbstractTensor) = u.yy
@inline _yz(u::AbstractTensor) = u.yz
@inline _zz(u::AbstractTensor) = u.zz

@inline Base.map(f::F, vecs::Vararg{AbstractSymmetricTensor{2}}) where {F <: Function} = @inline(SymmetricTensor(f(_xx.(vecs)...), f(_xy.(vecs)...), f(_xz.(vecs)...),
                                                                                                                 f(_yy.(vecs)...), f(_yz.(vecs)...), f(_zz.(vecs)...)))

@inline Base.map(f::F, vecs::Vararg{AbstractSymmetricTensor{N}}) where {F <: Function, N} = @inline(SymmetricTensor(map(f, _xx.(vecs)...), map(f, _xy.(vecs)...), map(f, _xz.(vecs)...),
                                                                                                                    map(f, _yy.(vecs)...), map(f, _yz.(vecs)...), map(f, _zz.(vecs)...)))

@inline Base.sum(v::AbstractSymmetricTensor{2}) = muladd(2, v.xy + v.xz + v.yz, v.xx + v.yy + v.zz)

@inline Base.sum(v::AbstractSymmetricTensor) = muladd(2, sum(v.xy) + sum(v.xz) + sum(v.yz), sum(v.xx) + sum(v.yy) + sum(v.zz))

@inline Base.sum(op::F, v::AbstractSymmetricTensor{2}) where {F <: Function} = muladd(2, op(v.xy) + op(v.xz) + op(v.yz), op(v.xx) + op(v.yy) + op(v.zz))

@inline Base.sum(op::F, v::AbstractSymmetricTensor) where {F <: Function} = muladd(2, sum(op, v.xy) + sum(op, v.xz) + sum(op, v.yz), sum(op, v.xx) + sum(op, v.yy) + sum(op, v.zz))

@inline Base.maximum(v::AbstractSymmetricTensor{2}) = max(sym_ten_fields(v)...)

@inline Base.maximum(v::AbstractSymmetricTensor) = max(maximum(v.xx), maximum(v.xy), maximum(v.xz),
                                                                      maximum(v.yy), maximum(v.yz),
                                                                                     maximum(v.zz))

@inline Base.maximum(op::F, v::AbstractSymmetricTensor{2}) where {F <: Function} = max(map(op, sym_ten_fields(v))...)

@inline Base.maximum(op::F, v::AbstractSymmetricTensor) where {F <: Function} = max(maximum(op, v.xx), maximum(op, v.xy), maximum(op, v.xz),
                                                                                                       maximum(op, v.yy), maximum(op, v.yz),
                                                                                                                          maximum(op, v.zz))

@inline Base.minimum(v::AbstractSymmetricTensor{2}) = min(sym_ten_fields(v)...)

@inline Base.minimum(v::AbstractSymmetricTensor) = min(minimum(v.xx), minimum(v.xy), minimum(v.xz),
                                                                      minimum(v.yy), minimum(v.yz),
                                                                                     minimum(v.zz))

@inline Base.minimum(op::F, v::AbstractSymmetricTensor{2}) where {F <: Function} = min(map(op, sym_ten_fields(v))...)

@inline Base.minimum(op::F, v::AbstractSymmetricTensor) where {F <: Function} = min(minimum(op, v.xx), minimum(op, v.xy), minimum(op, v.xz),
                                                                                                       minimum(op, v.yy), minimum(op, v.yz),
                                                                                                                          minimum(op, v.zz))

@inline Base.any(f::F, v::AbstractSymmetricTensor{2}) where {F <: Function} = f(v.xx) || f(v.xy) || f(v.xz) || f(v.yy) || f(v.yz) || f(v.zz)

@inline Base.any(f::F, v::AbstractSymmetricTensor) where {F <: Function} = any(f, v.xx) || any(f, v.xy) || any(f, v.xz) || any(f, v.yy) || any(f, v.yz) || any(f, v.zz)

@inline Base.all(f::F, v::AbstractSymmetricTensor{2}) where {F <: Function} = f(v.xx) && f(v.xy) && f(v.xz) && f(v.yy) && f(v.yz) && f(v.zz)

@inline Base.all(f::F, v::AbstractSymmetricTensor) where {F <: Function} = all(f, v.xx) && all(f, v.xy) && all(f, v.xz) && all(f, v.yy) && all(f, v.yz) && all(f, v.zz)

@inline Base.reverse(v::Vec) = Vec(v.z, v.y, v.x)

@inline Base.ifelse(b::Bool, T1::AbstractTensor{N}, T2::AbstractTensor{N}) where {N} = Tensor(ifelse(b, T1.x, T2.x), ifelse(b, T1.y, T2.y), ifelse(b, T1.z, T2.z))

@inline Base.ifelse(b::Bool, S1::AbstractSymmetricTensor{N}, S2::AbstractSymmetricTensor{N}) where {N} =
    SymmetricTensor(ifelse(b, S1.xx, S2.xx), ifelse(b, S1.xy, S2.xy), ifelse(b, S1.xz, S2.xz),ifelse(b, S1.yy, S2.yy),ifelse(b, S1.yz, S2.yz),ifelse(b, S1.zz, S2.zz))

@inline function Base.sort(v::Vec)
    _a = v.x
    _b = v.y
    _c = v.z

    lo = min(_a,_b)
    hi = max(_a,_b)
    a, b = lo, hi

    lo = min(b,_c)
    hi = max(b,_c)
    b, c = lo, hi

    lo = min(a,b)
    hi = max(a,b)
    a, b = lo, hi

    return Vec(a, b, c)
end

#### AbstractMatrix interface defined in the `Base` module #######

@inline function Base.transpose(T::Ten)
    x = T.x
    y = T.y
    z = T.z
    nx = Tensor(x.x, y.x, z.x)
    ny = Tensor(x.y, y.y, z.y)
    nz = Tensor(x.z, y.z, z.z)
    return Tensor(nx, ny, nz)
end

@inline Base.transpose(S::SymTen) = S

@inline Base.transpose(W::AntiSymTen) = -W

@inline Base.adjoint(T::Ten) = conj(transpose(T))

