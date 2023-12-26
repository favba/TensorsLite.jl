using VectorBasis
using Zeros
using Test

const ze = Zero()

@testset "Constructors" begin

    @test Vec{1}(1).v === 1
    @test Vec{1}(1.0).v === 1.0
    @test typeof(Vec(1.0)) === Vec1D{1,Float64}

    @test_throws DomainError Vec{0}(1)

    @test xx(Vec{1,2}(1,2)) === 1
    @test yy(Vec{1,2}(1,2)) === 2
    @test xx(Vec{1,2}(1.0,2.0)) === 1.0
    @test yy(Vec{1,2}(1.0,2.0)) === 2.0

    @test xx(Vec{2,1}(1.0,2.0)) === 2.0
    @test yy(Vec{2,1}(1.0,2.0)) === 1.0

    @test typeof(Vec(1.0,2)) === Vec2D{1,2,Float64}

    @test_throws DomainError Vec{0,2}(1.0,2.0)
    @test_throws DomainError Vec{1,4}(1.0,2.0)
    @test_throws DomainError Vec{1,1}(1.0,2.0)

    @test typeof(Vec(1,3,4.0im)) === Vec3D{ComplexF64}

end

@testset "Size and Length" begin
end

@testset "getindex" begin

    @test all(x->(x===ze),Vec())

    a1 = Vec1D{1}(1.0)
    @test a1[1] === 1.0
    @test a1[2] === ze
    @test a1[3] === ze

    a2 = Vec1D{2}(1.0)
    @test a2[1] === ze
    @test a2[2] === 1.0
    @test a2[3] === ze

    a3 = Vec1D{3}(1.0)
    @test a3[1] === ze
    @test a3[2] === ze
    @test a3[3] === 1.0

end
