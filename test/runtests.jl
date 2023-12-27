using VectorBasis
using Zeros
using Test

const ze = Zero()

@testset "Constructors" begin

    @test Vec(x=1).x === 1
    @test Vec(y=1.0).y === 1.0
    @test typeof(Vec(z=1.0)) === Vec{Union{Zero,Float64},1,Zero,Zero,Float64}

    @test eltype(Vec(x=1.0,y=2)) === Union{Zero,Float64}

    @test eltype(Vec(x=1,y=3,z=4.0im)) === ComplexF64

end

@testset "Size and Length" begin
end

@testset "getindex" begin

    @test all(x->(x===ze),Vec())

    a1 = Vec(x=1.0)
    @test a1[1] === 1.0
    @test a1[2] === ze
    @test a1[3] === ze

    a2 = Vec(y=1.0)
    @test a2[1] === ze
    @test a2[2] === 1.0
    @test a2[3] === ze

    a3 = Vec(z=1.0)
    @test a3[1] === ze
    @test a3[2] === ze
    @test a3[3] === 1.0

end
