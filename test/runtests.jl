using TensorsLite
using Zeros
using Test
using LinearAlgebra
using StrideArraysCore: object_and_preserve
import SIMD

const ze = Zero()

# This helps comparing arrays of Union{Zero,Number}
(::Type{<:Union{Zero, T}})(x::Number) where {T <: Number} = x == zero(x) ? Zeros.Zero() : T(x)

Base.isapprox(a::SIMD.Vec{N}, b::SIMD.Vec{N}) where {N} = reduce(&, ntuple(i -> isapprox(a[i], b[i]), Val{N}()))

const zeroSIMD = SIMD.Vec(0.0, 0.0, 0.0, 0.0)

my_isapprox(x, y) = isapprox(x, y)
my_isapprox(x::SIMD.Vec{N, T}, y::Number) where {N, T} = isapprox(x, SIMD.Vec{N, T}(y))
my_isapprox(y::Number, x::SIMD.Vec{N, T}) where {N, T} = isapprox(x, SIMD.Vec{N, T}(y))

@testset "_muladd definitions" begin
    for x in (One(), Zero(), rand(), SIMD.Vec(rand(), rand(), rand(), rand()))
        for y in (One(), Zero(), rand(), SIMD.Vec(rand(), rand(), rand(), rand()))
            for z in (One(), Zero(), rand(), SIMD.Vec(rand(), rand(), rand(), rand()))
                if (all(t -> typeof(t) === Float64, (x, y, z)))
                    @test TensorsLite._muladd(x, y, z) === muladd(x, y, z)
                else
                    @test my_isapprox(TensorsLite._muladd(x, y, z), TensorsLite.:+(TensorsLite.:*(x, y), z))
                end
            end
        end
    end
    @test TensorsLite._muladd(Zero(), 1.0im, 2.0) === 2.0 + 0.0im
    @test TensorsLite._muladd(1.0im, Zero(), 2.0) === 2.0 + 0.0im
end

const sx = SIMD.Vec(1.0, 2.0)
const sy = SIMD.Vec(2.0, 1.0)
const sz = SIMD.Vec(3.0, 4.0)

@testset "Vec Constructors" begin

    @test Vec1Dx(1).x === 1
    @test Vec1Dy(1.0).y === 1.0
    @test typeof(Vec1Dz(1.0)) === Vec1Dz{Float64}

    @test eltype(Vec2Dxy(1.0,2)) === Union{Zero, Float64}

    @test eltype(Vec(1, 3, 4.0im)) === ComplexF64

    @test eltype(Vec1Dx(One())) === Union{Zero, One}

    @test Vec1Dx(sx).x === sx
    @test Vec1Dy(sy).y === sy
    @test typeof(Vec1Dz(sz)) === Vec1Dz{SIMD.Vec{2, Float64}}

    @test eltype(Vec2Dxy(sx, sy)) === Union{Zero, SIMD.Vec{2, Float64}}

end

@testset "Tensor Constructors" begin
    @test Ten(
        xx = 1, xy = 2, xz = 3,
        yx = 4, yy = 5, yz = 6,
        zx = 7, zy = 8, zz = 9.0
    ) == [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]

    @test eltype(
        Ten(
            xx = 1, xy = 2, xz = 3,
            yx = 4, yy = 5, yz = 6,
            zx = 7, zy = 8, zz = 9.0
        )
    ) === Float64

    @test eltype(Ten(xx = 1.0)) === Union{Zero, Float64}

    @test eltype(Ten(xx = sx)) === Union{Zero, SIMD.Vec{2, Float64}}

    @test eltype(Ten(xx = One())) === Union{Zero, One}

    @test Vec{2}() === Ten()
    @test Vec2Dxy(Vec2Dxy(1.0, 2.0), Vec2Dxy(3.0, 4.0)).z === Vec()

    @test typeof(Vec2Dxz(Vec2Dxz(1.0,3.0), Vec2Dxz(-1, 3))) === Ten2Dxz{Float64}
    @test typeof(Vec2Dyz(Vec2Dyz(1.0,3.0), Vec2Dyz(-1, 3))) === Ten2Dyz{Float64}

    @test typeof(Vec1Dx(Vec1Dx(1.0))) === Ten1Dx{Float64}
    @test typeof(Vec1Dy(Vec1Dy(1.0))) === Ten1Dy{Float64}
    @test typeof(Vec1Dz(Vec1Dz(1.0))) === Ten1Dz{Float64}

    @test Ten3D(1,2,3,4,5,6,7,8,9) === Ten(1,2,3,4,5,6,7,8,9)
    @test Ten2Dxy(1,2,3,4.) === Ten(xx=1.0, xy=2.0, yx=3.0, yy=4.0)
    @test Ten2Dxz(1,2,3,4.) === Ten(xx=1.0, xz=2.0, zx=3.0, zz=4.0)
    @test Ten2Dyz(1,2,3,4.) === Ten(yy=1.0, yz=2.0, zy=3.0, zz=4.0)

    @test Ten1Dx(1) === Ten(xx=1)
    @test Ten1Dy(1) === Ten(yy=1)
    @test Ten1Dz(1) === Ten(zz=1)

end

@testset "Vec size and length" begin
    let a = Vec(rand(), rand(), rand()), T = Ten2Dyz(rand(), rand(), rand(), rand())
        @test size(a) === (3,)
        @test length(a) === 3
        @test size(T) === (3, 3)
        @test length(T) === 9
    end
end

@testset "Vec getindex" begin

    @test all(x -> (x === ze), Vec())

    let a1 = Vec1Dx(1.0)
        @test a1[1] === 1.0
        @test a1[2] === ze
        @test a1[3] === ze
    end

    let a2 = Vec1Dy(1.0)
        @test a2[1] === ze
        @test a2[2] === 1.0
        @test a2[3] === ze
    end

    let a3 = Vec1Dz(1.0)
        @test a3[1] === ze
        @test a3[2] === ze
        @test a3[3] === 1.0
    end

    let T = Ten(
            xx = 1, xy = 2, xz = 3,
            yx = 4, yy = 5, yz = 6,
            zx = 7, zy = 8, zz = 9.0
        )
        @test T[1, 1] === 1.0
        @test T[1, 2] === 2.0
        @test T[1, 3] === 3.0
        @test T[2, 1] === 4.0
        @test T[2, 2] === 5.0
        @test T[2, 3] === 6.0
        @test T[3, 1] === 7.0
        @test T[3, 2] === 8.0
        @test T[3, 3] === 9.0
    end

end

_rand(T) = rand(T)
_rand(::Type{Int64}) = rand((1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

@testset "Vector Operations" begin

    @test Vec(1, 2, 3) + [1, 2, 3] == [2, 4, 6]
    @test [1, 2, 3] + Vec(1, 2, 3) == [2, 4, 6]
    @test Vec(3, 2, 1) - [1, 2, 3] == [2, 0, -2]
    @test [1, 2, 3] - Vec(3, 2, 1) == [-2, 0, 2]

    @test convert(Vec3D{Float64}, 𝐢) === Vec(1.0, 0.0, 0.0)
    @test convert(Vec3D{Float64}, 𝐤) === Vec(0.0, 0.0, 1.0)
    @test convert(Vec2Dxy{Float64}, 𝐢) === Vec2Dxy(1.0, 0.0)
    @test convert(Vec2Dxz{Float64}, 𝐤) === Vec2Dxz(0.0, 1.0)

    @test zero(1.0𝐢) === Vec1Dx(0.0)
    @test zero(1𝐣) === Vec1Dy(0)
    @test zero(Vec2Dyz(1.0, im)) === Vec2Dyz(zero(1.0im), zero(1.0im))

    @test Vec(1, 2, 3) // 4 === Vec(1 // 4, 1 // 2, 3 // 4)

    @test +(Vec(1, 1, 1), Vec(1, 1, 1), Vec(1, 1, 1), Vec(1, 1, 1)) === Vec(4, 4, 4)

    @test norm(-2.0𝐢) === 2.0
    @test norm(-2.0𝐤) === 2.0
    @test norm(Vec()) === Zero()

    @test normalize(Vec()) === Vec()

    @test Vec1Dx(1.0) ≈ Vec2Dxy(1.0, eps())
    @test isapprox(Vec1Dx(1.0), Vec2Dxy(1.0, eps()); rtol = sqrt(eps()))

    for T1 in (Int64, Float64, ComplexF64)
        un = (Vec1Dy(_rand(T1)), Vec2Dxz(_rand(T1), _rand(T1)), Vec(_rand(T1), _rand(T1), _rand(T1)))
        for u in un
            Au = Array{nonzero_eltype(u)}(u)
            for op in (+, -, normalize, sum, x -> sum(exp, x), x -> map(exp, x))
                @test op(u) ≈ op(Au)
            end
            @test norm(u) ≈ norm(Au)
            if T1 === ComplexF64
                @test conj(u) == conj(Au)
            end
        end
        for T2 in (Int64, Float64, ComplexF64)
            vn = (Vec1Dy(_rand(T2)), Vec2Dxz(_rand(T2), _rand(T2)), Vec(_rand(T2), _rand(T2), _rand(T2)))
            for u in un
                Au = Array{nonzero_eltype(u)}(u)
                for v in vn
                    Av = Array{nonzero_eltype(v)}(v)
                    for op in (+, -, cross)
                        @test op(u, v) ≈ op(Au, Av)
                    end
                    @test dot(u, v) ≈ dot(conj(Au), Av) # Use conj here because for complex vectors julia conjugates the first vector automatically and we don't do that.
                    @test dotadd(u, v, 3.0) ≈ dot(conj(Au), Av) + 3.0
                    @test muladd(2.0, u, v) ≈ (2.0 * Au + Av)
                    @test muladd(u, 2.0, v) ≈ (2.0 * Au + Av)
                    @test inner(u, v) ≈ dot(Au, Av)
                    @test u ⊗ v ≈ Au * transpose(Av)
                end
            end
        end
    end
end

@testset "Tensor Operations" begin

    @test Ten1Dx(1.0) ≈ (1.0(𝐢⊗𝐢) + eps()*(𝐢⊗𝐣)) 
    @test isapprox(Ten1Dx(1.0), (1.0(𝐢⊗𝐢) + eps()*(𝐢⊗𝐣)); rtol = sqrt(eps()))

    for T1 in (Int64, Float64, ComplexF64)
        un = (
            Ten1Dy(_rand(T1)),
            Ten2Dxz(_rand(T1), _rand(T1), _rand(T1), _rand(T1)),
            Ten(
                _rand(T1), _rand(T1), _rand(T1),
                _rand(T1), _rand(T1), _rand(T1),
                _rand(T1), _rand(T1), _rand(T1)
            ),
        )
        for u in un
            Au = Array{nonzero_eltype(u)}(u)
            for op in (+, -, normalize, sum, x -> sum(exp, x), x -> map(exp, x))
                @test op(u) ≈ op(Au)
            end
            @test norm(u) ≈ norm(Au)
        end
        for T2 in (Int64, Float64, ComplexF64)
            vn = (
                Ten1Dy(_rand(T1)),
                Ten2Dxz(_rand(T1), _rand(T1), _rand(T1), _rand(T1)),
                Ten(
                    _rand(T1), _rand(T1), _rand(T1),
                    _rand(T1), _rand(T1), _rand(T1),
                    _rand(T1), _rand(T1), _rand(T1)
                ),
            )
            for u in un
                Au = Array{nonzero_eltype(u)}(u)
                for v in vn
                    Av = Array{nonzero_eltype(v)}(v)
                    for op in (+, -, *)
                        @test op(u, v) ≈ op(Au, Av)
                    end
                    @test dot(u, v) ≈ Au * Av
                    @test muladd(2.0, u, v) ≈ (2.0 * Au + Av)
                    @test muladd(u, 2.0, v) ≈ (2.0 * Au + Av)

                    @test muladd(u, v, v) ≈ (Au * Av + Av)
                    @test dotadd(u, v, v) ≈ (Au * Av + Av)
                    @test inner(u, v) ≈ dot(Au, Av)
                end
            end
        end
    end
end

@testset "Tensor x Vec Operations" begin
    for T1 in (Int64, Float64, ComplexF64)
        Tn = (
            Ten1Dy(_rand(T1)),
            Ten2Dxz(_rand(T1), _rand(T1), _rand(T1), _rand(T1)),
            Ten(
                _rand(T1), _rand(T1), _rand(T1),
                _rand(T1), _rand(T1), _rand(T1),
                _rand(T1), _rand(T1), _rand(T1)
            ),
        )

        for T2 in (Int64, Float64, ComplexF64)
            vn = (Vec1Dy(_rand(T2)), Vec2Dxz(_rand(T2), _rand(T2)), Vec(_rand(T2), _rand(T2), _rand(T2)))
            for T in Tn
                AT = Array{nonzero_eltype(T)}(T)

                @test transpose(T) == transpose(AT)
                @test T' == AT'

                for v in vn
                    Av = Array{nonzero_eltype(v)}(v)

                    @test T * v ≈ AT * Av
                    @test v * T ≈ transpose(AT) * Av
                    @test muladd(T, v, v) ≈ (AT * Av + Av)
                    @test dotadd(T, v, v) ≈ (AT * Av + Av)
                    @test muladd(v, T, v) ≈ (transpose(AT) * Av + Av)
                    @test dotadd(v, T, v) ≈ (transpose(AT) * Av + Av)
                    @test dot(v, T, v) ≈ dot(conj(Av), AT * Av)
                end
            end
        end
    end
end

@testset "SymTen" begin
    @test typeof(SymTen(1, 2, 3, 4, 5, 6)) === SymTen3D{Int}
    @test typeof(SymTen(xx = 1.0, xy = 2, yy = 3)) === SymTen2Dxy{Float64}
    @test typeof(SymTen(xx = 1.0, xz = 2, zz = 3)) === SymTen2Dxz{Float64}
    @test typeof(SymTen(yy = 1.0, yz = 2, zz = 3)) === SymTen2Dyz{Float64}

    @test SymTen3D(1,2,3,4,5,6.) === SymTen(xx=1.0,xy=2.0,xz=3.0,yy=4.0,yz=5.0,zz=6.0)
    @test SymTen2Dxy(1,2,3.) === SymTen(xx=1.0,xy=2.0,yy=3.0)
    @test SymTen2Dxz(1,2,3.) === SymTen(xx=1.0,xz=2.0,zz=3.0)
    @test SymTen2Dyz(1,2,3.) === SymTen(yy=1.0,yz=2.0,zz=3.0)
    @test SymTen1Dx(1) === SymTen(xx=1)
    @test SymTen1Dy(1) === SymTen(yy=1)
    @test SymTen1Dz(1) === SymTen(zz=1)

    @test -SymTen(1, 2, 3, 4, 5, 6) === SymTen(-1, -2, -3, -4, -5, -6)

    @test SymTen(1, 2, 3, 4, 5, 6) == [
        1 2 3;
        2 4 5;
        3 5 6
    ]

    @test SymTen(6, 5, 4, 3, 2, 0) == SymTen(6.0, 5.0, 4.0, 3.0, 2.0, 0.0)

    @test (SymTen(xx = 1) + SymTen(xx = 2.0, yy = 5.0)) === SymTen(xx = 3.0, yy = 5.0)
    @test (SymTen(xx = 1) - SymTen(xx = 2.0, yy = 5.0)) === SymTen(xx = -1.0, yy = -5.0)
    @test muladd(3.0, SymTen(xx = 4.0), SymTen(xx = 5.0)) === SymTen(xx = 17.0)
    @test muladd(One(), SymTen(1, 2, 3, 4, 5, 6), SymTen()) === SymTen(1, 2, 3, 4, 5, 6)

    let xx = rand(ComplexF64),yx = rand(ComplexF64),zx = rand(ComplexF64),yy = rand(ComplexF64),zy = rand(ComplexF64),zz = rand(ComplexF64)
        S = SymTen(xx, yx, zx, yy, zy, zz)
        @test S.x === Vec(xx, yx, zx)
        @test S.y === Vec(yx, yy, zy)
        @test S.z === Vec(zx, zy, zz)
        @test transpose(S) === S
        @test S' === SymTen(conj(xx), conj(yx), conj(zx), conj(yy), conj(zy), conj(zz))
        @test S.xy === S.yx
        @test S.zy === S.yz
        @test S.zx === S.xz
    end

    @test convert(SymTen2Dxy{Float64}, SymTen(xx = 1)) === SymTen(xx = 1.0, xy = 0.0, yy = 0.0)
    @test convert(Ten2Dxz{Float64}, SymTen(xz = 1)) === Ten(xx = 0.0, zx = 1.0, xz = 1.0, zz = 0.0)

    S1 = SymTen(rand(),rand(),rand(),rand(),rand(),rand())
    S2 = SymTen(rand(),rand(),rand(),rand(),rand(),rand())
    @test inner(S1, S2) ≈ dot(Array(S1), Array(S2))
end

@testset "AntiSymTen" begin
    @test typeof(AntiSymTen(xy = 1.0, xz = 2, yz = 3)) === AntiSymTen3D{Float64}
    @test typeof(AntiSymTen(xy = 1.0)) === AntiSymTen2Dxy{Float64}
    @test typeof(AntiSymTen(yz = 2)) === AntiSymTen2Dyz{Int}
    @test typeof(AntiSymTen(xz = 2.0)) === AntiSymTen2Dxz{Float64}

    @test AntiSymTen3D(1,2,3.0) === AntiSymTen(xy=1.0, xz=2.0, yz=3.0)
    @test AntiSymTen2Dxy(1) === AntiSymTen(xy=1)
    @test AntiSymTen2Dxz(1) === AntiSymTen(xz=1)
    @test AntiSymTen2Dyz(1) === AntiSymTen(yz=1)

    @test -AntiSymTen(1, 2, 3) === AntiSymTen(-1, -2, -3)

    @test AntiSymTen(1, 2, 3) == [
         0   1  2;
        -1   0  3;
        -2  -3  0
    ]

    @test AntiSymTen(3, 2, 1) == AntiSymTen(3.0, 2.0, 1.0)

    @test (AntiSymTen(xy = 1) + AntiSymTen(xy = 2.0, xz = 5.0)) === AntiSymTen(xy = 3.0, xz = 5.0)
    @test (AntiSymTen(xy = 1) - AntiSymTen(xy = 2.0, xz = 5.0)) === AntiSymTen(xy = -1.0, xz = -5.0)
    @test muladd(3.0, AntiSymTen(xy = 4.0), AntiSymTen(xy = 5.0)) === AntiSymTen(xy = 17.0)
    @test muladd(One(), AntiSymTen(1, 2, 3), AntiSymTen()) === AntiSymTen(1, 2, 3)

    let xy = rand(ComplexF64), xz = rand(ComplexF64), yz = rand(ComplexF64)
        W = AntiSymTen(xy, xz, yz)
        @test W.x === Vec2Dyz(-xy, -xz)
        @test W.y === Vec2Dxz(xy, -yz)
        @test W.z === Vec2Dxy(xz, yz)
        @test transpose(W) === -W
        @test W' === AntiSymTen(-conj(xy), -conj(xz), -conj(yz))
    end

    @test convert(AntiSymTen3D{Float64}, AntiSymTen(xy = 1)) === AntiSymTen(xy = 1.0, xz = 0.0, yz = 0.0)
    @test convert(Ten2Dxz{Float64}, AntiSymTen(xz = 1)) === Ten(xx = 0.0, xz = 1.0, zx = -1.0, zz = 0.0)

    W1 = AntiSymTen(rand(), rand(), rand())
    W2 = AntiSymTen(rand(), rand(), rand())

    @test inner(W1, W2) ≈ dot(Array(W1), Array(W2))

end

@testset "VecArray" begin
    #@test_throws DomainError VecArray() Now creates a Zero Dimensional Vec3DArray{Zero, 0}
    @test_throws DimensionMismatch Vec2DxyArray(rand(1, 2), rand(2, 1))
    @test_throws DimensionMismatch Vec2DxzArray(rand(1, 2), rand(2, 1))
    @test_throws DimensionMismatch Vec2DyzArray(rand(1, 2), rand(2, 1))

    @test size(VecArray{Float32}(4, 3)) === (4, 3)
    @test eltype(VecArray{Float32}(4, 3)) === Vec3D{Float32}

    @test Vec1DxArray(ones(1, 1))[1] === Vec1Dx(1.0)
    @test Vec1DyArray(ones(1, 1))[1] === Vec1Dy(1.0)
    @test Vec1DzArray(ones(1, 1))[1] === Vec1Dz(1.0)

    @test Vec2DxyArray(ones(1, 1), (ones(1, 1) .+ 1))[1] === Vec2Dxy(1.0, 2.0)
    @test Vec2DyzArray(ones(1, 1), (ones(1, 1) .+ 1))[1] === Vec2Dyz(1.0, 2.0)
    @test Vec2DxzArray(ones(1, 1), (ones(1, 1) .+ 1))[1] === Vec2Dxz(1.0, 2.0)
    @test VecArray(ones(1, 1), (ones(1, 1) .+ 1), (ones(1, 1) .+ 2))[1] === Vec(1.0, 2.0, 3.0)

    @test Vec1DxArray(ones(1, 1))[1, 1] === Vec1Dx(1.0)
    @test Vec1DyArray(ones(1, 1))[1, 1] === Vec1Dy(1.0)
    @test Vec1DzArray(ones(1, 1))[1, 1] === Vec1Dz(1.0)

    @test Vec2DxyArray(ones(1, 1), (ones(1, 1) .+ 1))[1, 1] === Vec2Dxy(1.0, 2.0)
    @test Vec2DyzArray(ones(1, 1), (ones(1, 1) .+ 1))[1, 1] === Vec2Dyz(1.0, 2.0)
    @test Vec2DxzArray(ones(1, 1), (ones(1, 1) .+ 1))[1, 1] === Vec2Dxz(1.0, 2.0)
    @test VecArray(ones(1, 1), (ones(1, 1) .+ 1), (ones(1, 1) .+ 2))[1, 1] === Vec(1.0, 2.0, 3.0)


    @test setindex!(VecArray{Float64}(4, 4), Vec(1.0, 2.0, 3.0), 4, 4)[4, 4] === Vec(1.0, 2.0, 3.0)
    @test setindex!(VecArray{Float64}(4, 4), Vec(1.0, 2.0, 3.0), 16)[16] === Vec(1.0, 2.0, 3.0)

    @test setindex!(Vec1DyArray(zeros(4, 4)), Vec(0.0, 2.0, 0.0), 4, 4)[4, 4] === Vec1Dy(2.0)
    @test setindex!(Vec1DyArray(zeros(4, 4)), Vec(0.0, 2.0, 0.0), 16)[16] === Vec1Dy(2.0)

    let a = Vec2DxyArray(rand(Float32, 2), rand(Float32, 2))
        @test typeof(similar(a, Vec2Dyz{Int}, (4, 4, 4))) === Vec2DyzArray{Int, 3}
        @test a == object_and_preserve(a)[1]
        b = Vec2DxyArray(a.y, rand(Float32,2))
        @test Base.mightalias(a, b)
        @test length(resize!(a, 4)) === 4
    end

    let a = Ten2DxzArray(rand(2), rand(2), rand(2), rand(2))
        @test typeof(similar(a, Ten3D{Float16}, (1, 1, 1))) === Ten3DArray{Float16, 3}
        @test a == object_and_preserve(a)[1]
        b = Ten2DyzArray(a.zx, rand(2), rand(2), rand(2))
        @test Base.mightalias(a,b)
        @test length(resize!(a, 4)) === 4
    end

    let xx = [1],xy = [2],xz = [3],yx = [4],yy = [5],yz = [6],zx = [7],zy = [8],zz = [9],T = TenArray(xx, xy, xz,
                                                                                                      yx, yy, yz,
                                                                                                      zx, zy, zz)
        @test T.xx === xx
        @test T.yx === yx
        @test T.zx === zx
        @test T.xy === xy
        @test T.yy === yy
        @test T.zy === zy
        @test T.xz === xz
        @test T.yz === yz
        @test T.zz === zz

        @test Vec3DArray(VecArray(xx,yx,zx), VecArray(xy, yy, zy), VecArray(xz, yz, zz)) === T
        @test Vec2DxyArray(Vec2DxyArray(xx,yx), Vec2DxyArray(xy, yy)) == Ten2DxyArray(xx, xy, yx, yy)
        @test Vec2DxzArray(Vec2DxzArray(xx,zx), Vec2DxzArray(xz, zz)) == Ten2DxzArray(xx, xz, zx, zz)
        @test Vec2DyzArray(Vec2DyzArray(yy,zy), Vec2DyzArray(yz, zz)) == Ten2DyzArray(yy, yz, zy, zz)

        @test Vec1DxArray(Vec1DxArray(xx)) == Ten1DxArray(xx)
        @test Vec1DyArray(Vec1DyArray(yy)) == Ten1DyArray(yy)
        @test Vec1DzArray(Vec1DzArray(zz)) == Ten1DzArray(zz)
    end

end

@testset "VecArray Broadcasting" begin
    let ux = Vec1DxArray(rand(3)), uxy = Vec2DxyArray(rand(Float32, 3), rand(Float32, 3)), uxyz = VecArray(rand(Float32, 3), rand(Float32, 3), rand(Float32, 3))
        @test (ux .+ uxy) == Vec2DxyArray(ux.x .+ uxy.x, Float64.(uxy.y))
        @test typeof(ux .+ uxy) === Vec2DxyArray{Float64, 1}
        @test (uxy .+ uxyz) == VecArray(uxy.x .+ uxyz.x, uxy.y .+ uxyz.y, uxyz.z)
        @test typeof(uxy .+ uxyz) === Vec3DArray{Float32, 1}
        @test ux .+ 𝐢 == Vec1DxArray(ux.x .+ 1)
        @test (ux .= Vec()) == Vec1DxArray(zeros(3))
    end
end

@testset "TenArray Broadcasting" begin
    let ux = TenArray(xx = rand(3)), uxy = TenArray(xx = rand(Float32, 3), yx = rand(Float32, 3), xy = rand(Float32, 3), yy = rand(Float32, 3))
        @test (ux .+ uxy) == Ten2DxyArray(ux.x.x .+ uxy.x.x, uxy.y.x, uxy.x.y, uxy.y.y)
        @test typeof(ux .+ uxy) === Ten2DxyArray{Float64, 1}
        @test ux .+ (𝐢 ⊗ 𝐢) == Ten1DxArray(ux.x.x .+ 1)
        @test (ux .= Ten()) == Ten1DxArray(zeros(3))
    end
end

@testset "SymTenArray" begin
    @test_throws DomainError SymTenArray()
    @test_throws DimensionMismatch SymTenArray(xx = rand(1, 2), xy = rand(2, 1))
    @test_throws DimensionMismatch SymTenArray(xx = rand(1, 2), xz = rand(2, 1))
    @test_throws DimensionMismatch SymTenArray(xx = rand(1, 2), yy = rand(2, 1))
    @test_throws DimensionMismatch SymTenArray(xx = rand(1, 2), zz = rand(2, 1))

    @test size(SymTenArray{Float32}(4, 3)) === (4, 3)
    @test eltype(SymTenArray{Float32}(4, 3)) === SymTen3D{Float32}

    @test eltype(SymTenArray(yy = rand(Int, 3), yz = rand(Int, 3), zz = rand(Int, 3))) === SymTen2Dyz{Int}

    @test typeof(similar(SymTenArray(yy = rand(Int, 3), yz = rand(Int, 3), zz = rand(Int, 3)), SymTen2Dxy{Float32})) === SymTen2DxyArray{Float32, 1}
    @test eltype(similar(SymTenArray(yy = rand(Int, 3), yz = rand(Int, 3), zz = rand(Int, 3)), SymTen2Dxy{Float32})) === SymTen2Dxy{Float32}
    @test size(similar(SymTenArray(yy = rand(Int, 3), yz = rand(Int, 3), zz = rand(Int, 3)), SymTen2Dxy{Float32}, 3, 4)) === (3, 4)

    let a1 = rand(3, 3),a2 = rand(3, 3),a3 = rand(3, 3),a4 = rand(3, 3),a5 = rand(3, 3),a6 = rand(3, 3),T = SymTenArray(a1, a2, a3,
                                                                                                                            a4, a5,
                                                                                                                                a6)
        @test T.xx === a1
        @test T.yx === a2
        @test T.zx === a3
        @test T.xy === a2
        @test T.yy === a4
        @test T.zy === a5
        @test T.xz === a3
        @test T.yz === a5
        @test T.zz === a6

        @test T == object_and_preserve(T)[1]
        @test Base.mightalias(SymTenArray(zz = T.yx), T)

        @test T.x === VecArray(a1, a2, a3)
        @test T.y === VecArray(a2, a4, a5)
        @test T.z === VecArray(a3, a5, a6)

        @test T[5] === SymTen(a1[5], a2[5], a3[5], a4[5], a5[5], a6[5])
        @test T[2, 3] === SymTen(a1[2, 3], a2[2, 3], a3[2, 3], a4[2, 3], a5[2, 3], a6[2, 3])

        @test setindex!(T, SymTen(xx = 1, yz = 3), 3)[3] === SymTen(1.0, 0.0, 0.0, 0.0, 3.0, 0.0)
        @test setindex!(T, SymTen(xy = 1, yy = 3), 3, 3)[3, 3] === SymTen(0.0, 1.0, 0.0, 3.0, 0.0, 0.0)

        @test typeof(T .+ SymTen()) === typeof(T)

        @test SymTen3DArray(a1,a2,a3,a4,a5,a6) === SymTenArray(a1,a2,a3,a4,a5,a6)
        @test SymTen2DxyArray(a1,a2,a3) == SymTenArray(xx=a1,xy=a2,yy=a3)
        @test SymTen2DxzArray(a1,a2,a3) == SymTenArray(xx=a1,xz=a2,zz=a3)
        @test SymTen2DyzArray(a1,a2,a3) == SymTenArray(yy=a1,yz=a2,zz=a3)
        @test SymTen1DxArray(a1) == SymTenArray(xx=a1)
        @test SymTen1DyArray(a1) == SymTenArray(yy=a1)
        @test SymTen1DzArray(a1) == SymTenArray(zz=a1)
    end
end

@testset "AntiSymTenArray" begin
    @test_throws DomainError AntiSymTenArray()
    @test_throws DimensionMismatch AntiSymTenArray(xy = rand(1, 2), yz = rand(2, 1))
    @test_throws DimensionMismatch AntiSymTenArray(xy = rand(1, 2), xz = rand(2, 1))

    @test size(AntiSymTenArray{Float32}(4, 3)) === (4, 3)
    @test eltype(AntiSymTenArray{Float32}(4, 3)) === AntiSymTen3D{Float32}

    @test eltype(AntiSymTenArray(yz = rand(Int, 3))) === AntiSymTen2Dyz{Int}

    @test typeof(similar(AntiSymTenArray(xy = rand(Int, 3)), AntiSymTen2Dxy{Float32})) === AntiSymTen2DxyArray{Float32, 1}
    @test eltype(similar(AntiSymTenArray(yz = rand(Int, 3)), AntiSymTen2Dxy{Float32})) === AntiSymTen2Dxy{Float32}
    @test size(similar(AntiSymTenArray(yz = rand(Int, 3)), AntiSymTen2Dxy{Float32}, 3, 4)) === (3, 4)

    let a1 = rand(3, 3),a2 = rand(3, 3),a3 = rand(3, 3), T = AntiSymTenArray(a1, a2, a3)
        zero_vec = Array{Zero}(undef, size(T))
        @test T.xx == zero_vec
        @test T.xy === a1
        @test T.xz === a2
        @test T.yx == -a1
        @test T.yy == zero_vec
        @test T.yz === a3
        @test T.zx == -a2
        @test T.zy == -a3
        @test T.zz == zero_vec

        @test T == object_and_preserve(T)[1]
        @test Base.mightalias(AntiSymTenArray(xz = T.xy), T)

        @test T.x == VecArray(zero_vec, -a1, -a2)
        @test T.y == VecArray(a1, zero_vec, -a3)
        @test T.z == VecArray(a2, a3, zero_vec)

        @test T[5] === AntiSymTen(a1[5], a2[5], a3[5])
        @test T[2, 3] === AntiSymTen(a1[2, 3], a2[2, 3], a3[2, 3])

        @test setindex!(T, AntiSymTen(yz = 3), 3)[3] === AntiSymTen(0.0, 0.0, 3.0)
        @test setindex!(T, AntiSymTen(xy = 1), 3, 3)[3, 3] === AntiSymTen(1.0, 0.0, 0.0)

        @test typeof(T .+ AntiSymTen()) === typeof(T)

        @test AntiSymTen3DArray(a1, a2, a3) === AntiSymTenArray(xy=a1,xz=a2,yz=a3)
        @test AntiSymTen2DxyArray(a1) == AntiSymTenArray(xy=a1)
        @test AntiSymTen2DxzArray(a1) == AntiSymTenArray(xz=a1)
        @test AntiSymTen2DyzArray(a1) == AntiSymTenArray(yz=a1)
    end
end

@testset "VecArray SIMD" begin
    I = SIMD.VecRange{4}(1)
    J = SIMD.Vec(2, 1, 4, 3)
    JF = SIMD.Vec(2.0, 1.0, 4.0, 3.0)
    u1D = Vec1DxArray(rand(4))
    @test u1D[I] === Vec1Dx(u1D.x[I])
    @test u1D[J] === Vec1Dx(u1D.x[J])
    @test begin
        u1D[I] = Vec1Dx(zeroSIMD)
        u1D[1:4] == Vec1DxArray( zeros(4))
        u1D[J] = Vec1Dx(JF)
        u1D[1:4] == Vec1DxArray(collect(1:4))
    end
    u2D = Vec2DxyArray(rand(4), rand(4))
    @test u2D[I] === Vec2Dxy(u2D.x[I], u2D.y[I])
    @test u2D[J] === Vec2Dxy(u2D.x[J], u2D.y[J])
    @test begin
        u2D[I] = Vec2Dxy(zeroSIMD, zeroSIMD)
        u2D[1:4] == Vec2DxyArray(zeros(4), zeros(4))
        u2D[J] = Vec2Dxy(JF, JF)
        u2D[1:4] == Vec2DxyArray(collect(1:4), collect(1:4))
    end
    u3D = VecArray(rand(4), rand(4), rand(4))
    @test u3D[I] === Vec(u3D.x[I], u3D.y[I], u3D.z[I])
    @test u3D[J] === Vec(u3D.x[J], u3D.y[J], u3D.z[J])
    @test begin
        u3D[I] = Vec(zeroSIMD, zeroSIMD, zeroSIMD)
        u3D[1:4] == VecArray(zeros(4), zeros(4), zeros(4))
        u3D[J] = Vec(JF, JF, JF)
        u3D[1:4] == VecArray(collect(1:4), collect(1:4), collect(1:4))
    end

    u1D2 = Vec1DxArray(rand(4, 2))
    @test u1D2[I, 2] === Vec1Dx(u1D2.x[I, 2])
    @test begin
        u1D2[I, 2] = Vec1Dx(zeroSIMD)
        u1D2[1:4, 2] == Vec1DxArray(zeros(4))
    end
    u2D2 = Vec2DxyArray(rand(4, 2), rand(4, 2))
    @test u2D2[I, 2] === Vec2Dxy(u2D2.x[I, 2], u2D2.y[I, 2])
    @test begin
        u2D2[I, 2] = Vec2Dxy(zeroSIMD, zeroSIMD)
        u2D2[1:4, 2] == Vec2DxyArray(zeros(4), zeros(4))
    end
    u3D2 = VecArray(rand(4, 2), rand(4, 2), rand(4, 2))
    @test u3D2[I, 2] === Vec(u3D2.x[I, 2], u3D2.y[I, 2], u3D2.z[I, 2])
    @test begin
        u3D2[I, 2] = Vec(zeroSIMD, zeroSIMD, zeroSIMD)
        u3D2[1:4, 2] == VecArray(zeros(4), zeros(4), zeros(4))
    end
end

@testset "TenArray SIMD" begin
    I = SIMD.VecRange{4}(1)
    u1D = TenArray(xx = rand(4))
    @test u1D[I] === Ten(xx = u1D.xx[I])
    @test begin
        u1D[I] = Ten(xx = zeroSIMD)
        u1D[1:4] == TenArray(xx = zeros(4))
    end
    u2D = TenArray(xx = rand(4), yx = rand(4), xy = rand(4), yy = rand(4))
    @test u2D[I] === Ten(xx = u2D.xx[I], yx = u2D.yx[I], xy = u2D.xy[I], yy = u2D.yy[I])
    @test begin
        u2D[I] = Ten(xx = zeroSIMD, yx = zeroSIMD, xy = zeroSIMD, yy = zeroSIMD)
        u2D[1:4] == TenArray(xx = zeros(4), yx = zeros(4), xy = zeros(4), yy = zeros(4))
    end
    u3D = TenArray(xx = rand(4), yx = rand(4), zx = rand(4), xy = rand(4), yy = rand(4), zy = rand(4), xz = rand(4), yz = rand(4), zz = rand(4))
    @test u3D[I] === Ten(xx = u3D.xx[I], yx = u3D.yx[I], zx = u3D.zx[I], xy = u3D.xy[I], yy = u3D.yy[I], zy = u3D.zy[I], xz = u3D.xz[I], yz = u3D.yz[I], zz = u3D.zz[I])
    @test begin
        u3D[I] = Ten(xx = zeroSIMD, yx = zeroSIMD, zx = zeroSIMD, xy = zeroSIMD, yy = zeroSIMD, zy = zeroSIMD, xz = zeroSIMD, yz = zeroSIMD, zz = zeroSIMD)
        u3D[1:4] == TenArray(xx = zeros(4), yx = zeros(4), zx = zeros(4), xy = zeros(4), yy = zeros(4), zy = zeros(4), xz = zeros(4), yz = zeros(4), zz = zeros(4))
    end

    u1D = TenArray(xx = rand(4, 2))
    @test u1D[I, 2] === Ten(xx = u1D.xx[I, 2])
    @test begin
        u1D[I, 2] = Ten(xx = zeroSIMD)
        u1D[1:4, 2] == TenArray(xx = zeros(4))
    end
    u2D = TenArray(xx = rand(4, 2), yx = rand(4, 2), xy = rand(4, 2), yy = rand(4, 2))
    @test u2D[I, 2] === Ten(xx = u2D.xx[I, 2], yx = u2D.yx[I, 2], xy = u2D.xy[I, 2], yy = u2D.yy[I, 2])
    @test begin
        u2D[I, 2] = Ten(xx = zeroSIMD, yx = zeroSIMD, xy = zeroSIMD, yy = zeroSIMD)
        u2D[1:4, 2] == TenArray(xx = zeros(4), yx = zeros(4), xy = zeros(4), yy = zeros(4))
    end
    u3D = TenArray(xx = rand(4, 2), yx = rand(4, 2), zx = rand(4, 2), xy = rand(4, 2), yy = rand(4, 2), zy = rand(4, 2), xz = rand(4, 2), yz = rand(4, 2), zz = rand(4, 2))
    @test u3D[I, 2] === Ten(xx = u3D.xx[I, 2], yx = u3D.yx[I, 2], zx = u3D.zx[I, 2], xy = u3D.xy[I, 2], yy = u3D.yy[I, 2], zy = u3D.zy[I, 2], xz = u3D.xz[I, 2], yz = u3D.yz[I, 2], zz = u3D.zz[I, 2])
    @test begin
        u3D[I, 2] = Ten(xx = zeroSIMD, yx = zeroSIMD, zx = zeroSIMD, xy = zeroSIMD, yy = zeroSIMD, zy = zeroSIMD, xz = zeroSIMD, yz = zeroSIMD, zz = zeroSIMD)
        u3D[1:4, 2] == TenArray(xx = zeros(4), yx = zeros(4), zx = zeros(4), xy = zeros(4), yy = zeros(4), zy = zeros(4), xz = zeros(4), yz = zeros(4), zz = zeros(4))
    end
end

function apply_simd_op(out, op::F, args::Vararg{Any, N}) where {F, N}
    l = length(out)
    @assert rem(l, 8) == 0
    lane = SIMD.VecRange{8}(0)
    @inbounds for i in 1:8:l
        lpi = lane + i
        out[lpi] = op(map(x -> getindex(x, lpi), args)...)
    end
    return out
end

@testset "SIMD Vec operations" begin

    let a = SIMD.Vec(2., 3., 4., 5.), v1D = 2.0𝐢, v3D = Vec(2.0, 3.0, 4.0)
        @test a * v1D === (a * v1D.x) * 𝐢
        @test a * v3D === Vec(a * v3D.x, a * v3D.y, a * v3D.z)
    end

    for u in (Vec1DxArray(rand(16)), Vec2DxyArray(rand(16), rand(16)), VecArray(rand(16), rand(16), rand(16)))

        for op in (+, -, norm, normalize, x -> (2 * x), x -> (x / 2))
            @test begin
                r = op.(u)
                rout = similar(r)
                all(map(isapprox, apply_simd_op(rout, op, u), r))
            end
        end

        for v in (Vec1DxArray(rand(16)), Vec2DxyArray(rand(16), rand(16)), VecArray(rand(16), rand(16), rand(16)))

            for op in (+, -, cross, dot, inner, (x, y) -> dotadd(x, y, 3.0), (x, y) -> muladd(2.0, x, y), (x, y) -> muladd(x, 2.0, y), ⊗)
                @test begin
                    r = op.(u, v)
                    rout = similar(r)
                    all(map(isapprox, apply_simd_op(rout, op, u, v), r))
                end
            end
        end
    end

end
