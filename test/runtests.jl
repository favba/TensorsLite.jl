using TensorsLite
using Zeros
using Test
using LinearAlgebra

const ze = Zero()

# This helps comparing arrays of Union{Zero,Number}
(::Type{<:Union{Zero,T}})(x::Number) where T<: Number = x == zero(x) ? Zeros.Zero() : T(x)

@testset "Vec Constructors" begin

    @test Vec(x=1).x === 1
    @test Vec(y=1.0).y === 1.0
    @test typeof(Vec(z=1.0)) === Vec1Dz{Float64}

    @test eltype(Vec(x=1.0,y=2)) === Union{Zero,Float64}

    @test eltype(Vec(x=1,y=3,z=4.0im)) === ComplexF64

    @test eltype(Vec(x=One())) === Union{Zero,One}

end

@testset "Tensor Constructors" begin
    @test Array(Ten(xx=1, xy=2, xz=3,
                    yx=4, yy=5, yz=6,
                    zx=7, zy=8, zz=9.0)) == [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]

    @test eltype(Ten(xx=1, xy=2, xz=3,
                     yx=4, yy=5, yz=6,
                     zx=7, zy=8, zz=9.0)) === Float64

    @test eltype(Ten(xx=1.0)) === Union{Zero,Float64}

    @test eltype(Ten(xx=One())) === Union{Zero,One}
end

@testset "Vec size and length" begin
    let a = Vec(rand(),rand(),rand()), T = Ten(xx=rand(),yy=rand(),xz=rand())
        @test size(a) === (3,)
        @test length(a) === 3
        @test size(T) === (3,3)
        @test length(T) === 9
    end
end

@testset "Vec getindex" begin

    @test all(x->(x===ze),Vec())

    let a1 = Vec(x=1.0)
        @test a1[1] === 1.0
        @test a1[2] === ze
        @test a1[3] === ze
    end

    let a2 = Vec(y=1.0)
        @test a2[1] === ze
        @test a2[2] === 1.0
        @test a2[3] === ze
    end

    let a3 = Vec(z=1.0)
        @test a3[1] === ze
        @test a3[2] === ze
        @test a3[3] === 1.0
    end

    let T = Ten(xx=1, xy=2, xz=3,
                yx=4, yy=5, yz=6,
                zx=7, zy=8, zz=9.0)
        @test T[1,1] === 1.0
        @test T[1,2] === 2.0
        @test T[1,3] === 3.0
        @test T[2,1] === 4.0
        @test T[2,2] === 5.0
        @test T[2,3] === 6.0
        @test T[3,1] === 7.0
        @test T[3,2] === 8.0
        @test T[3,3] === 9.0
    end

end

@testset "ZeroArray" begin

    @test size(ZeroArray(16,16,3)) === (16,16,3)
    zevec = ZeroArray(4,4)

    @test zevec[1] === ze
    @test zevec[4,4] === ze

    @test_throws InexactError setindex!(zevec,1.0,1)
    @test_throws InexactError setindex!(zevec,1.0,4,4)
end

@testset "VecArray" begin
    @test_throws DomainError VecArray()
    @test_throws DomainError VecArray(x=rand(1,2), y=rand(2,1))
    @test_throws DomainError VecArray(x=rand(1,2), z=rand(2,1))
    @test_throws DomainError VecArray(y=rand(1,2), z=rand(2,1))

    @test size(VecArray{Float32}(4,3)) === (4,3)
    @test eltype(VecArray{Float32}(4,3)) === Vec3D{Float32}

    @test VecArray(x=ones(1,1))[1] === Vec(x=1.0)
    @test VecArray(y=ones(1,1))[1] === Vec(y=1.0)
    @test VecArray(z=ones(1,1))[1] === Vec(z=1.0)

    @test VecArray(x=ones(1,1),y=(ones(1,1) .+ 1))[1] === Vec(x=1.0,y=2.0)
    @test VecArray(y=ones(1,1),z=(ones(1,1) .+ 1))[1] === Vec(y=1.0,z=2.0)
    @test VecArray(x=ones(1,1),z=(ones(1,1) .+ 1))[1] === Vec(x=1.0,z=2.0)
    @test VecArray(x=ones(1,1),y=(ones(1,1) .+ 1),z=(ones(1,1) .+ 2))[1] === Vec(x=1.0,y=2.0,z=3.0)

    @test VecArray(x=ones(1,1))[1,1] === Vec(x=1.0)
    @test VecArray(y=ones(1,1))[1,1] === Vec(y=1.0)
    @test VecArray(z=ones(1,1))[1,1] === Vec(z=1.0)

    @test VecArray(x=ones(1,1),y=(ones(1,1) .+ 1))[1,1] === Vec(x=1.0,y=2.0)
    @test VecArray(y=ones(1,1),z=(ones(1,1) .+ 1))[1,1] === Vec(y=1.0,z=2.0)
    @test VecArray(x=ones(1,1),z=(ones(1,1) .+ 1))[1,1] === Vec(x=1.0,z=2.0)
    @test VecArray(x=ones(1,1),y=(ones(1,1) .+ 1),z=(ones(1,1) .+ 2))[1,1] === Vec(x=1.0,y=2.0,z=3.0)


    @test setindex!(VecArray{Float64}(4,4),Vec(x=1.0,y=2.0,z=3.0),4,4)[4,4] === Vec(x=1.0,y=2.0,z=3.0)
    @test setindex!(VecArray{Float64}(4,4),Vec(x=1.0,y=2.0,z=3.0),16)[16] === Vec(x=1.0,y=2.0,z=3.0)

    @test setindex!(VecArray(y=zeros(4,4)),Vec(x=0.0,y=2.0,z=0.0),4,4)[4,4] === Vec(y=2.0)
    @test setindex!(VecArray(y=zeros(4,4)),Vec(x=0.0,y=2.0,z=0.0),16)[16] === Vec(y=2.0)

end

@testset "VecArray Broadcasting" begin
    let ux=VecArray(x=rand(3)), uxy=VecArray(x=rand(Float32,3),y=rand(Float32,3)), uxyz=VecArray(x=rand(Float32,3), y=rand(Float32,3), z=rand(Float32,3))
        @test (ux .+ uxy) == VecArray(x=ux.x .+ uxy.x, y=uxy.y)
        @test typeof(ux .+ uxy) === Vec2DxyArray{Float64,1}
        @test (uxy .+ uxyz) == VecArray(x=uxy.x .+ uxyz.x, y=uxy.y .+ uxyz.y, z=uxyz.z)
        @test typeof(uxy .+ uxyz) === Vec3DArray{Float32,1}
        @test ux .+ 𝐢 == VecArray(x=ux.x .+ 1)
        @test (ux .= Vec()) == VecArray(x=zeros(3))
    end
end

_rand(T) = rand(T)
_rand(T::Type{Int64}) = rand((1,2,3,4,5,6,7,8,9,10))
@testset "Vector Operations" begin

    @test Vec(1,2,3) + [1,2,3] == [2,4,6]
    @test [1,2,3] + Vec(1,2,3) == [2,4,6]
    @test Vec(3,2,1) - [1,2,3] == [2,0,-2]
    @test [1,2,3] - Vec(3,2,1) == [-2,0,2]
        

    for T1 in (Int64,Float64,ComplexF64)
        un = (Vec(y=_rand(T1)), Vec(x=_rand(T1), z=_rand(T1)), Vec(_rand(T1),_rand(T1),_rand(T1)))
        for u in un
            Au = Array{TensorsLite._my_eltype(u)}(u)
            for op in (+,-,normalize)
                @test op(u) ≈ op(Au)
            end
            @test norm(u) ≈ norm(Au)
            if T1 === ComplexF64
                @test conj(u) == conj(Au)
            end
        end
        for T2 in (Int64,Float64,ComplexF64)
            vn = (Vec(y=_rand(T2)), Vec(x=_rand(T2), z=_rand(T2)), Vec(_rand(T2),_rand(T2),_rand(T2)))
            for u in un
                Au = Array{TensorsLite._my_eltype(u)}(u)
                for v in vn
                    Av = Array{TensorsLite._my_eltype(v)}(v)
                    for op in (+,-,cross)
                        @test op(u,v) ≈ op(Au,Av)
                    end
                    @test dot(u,v) ≈ dot(conj(Au),Av) # Use conj here because for complex vectors julia conjugates the first vector automatically and we don't do that.
                    @test dotadd(u,v,3.0) ≈ dot(conj(Au),Av) + 3.0
                    @test muladd(2.0,u,v) ≈ (2.0*Au + Av)
                    @test muladd(u,2.0,v) ≈ (2.0*Au + Av)
                    @test inner(u,v) ≈ dot(Au,Av)
                    @test u⊗v ≈ Au*transpose(Av)
                end
            end
        end
    end
end

@testset "Tensor Operations" begin
    for T1 in (Int64,Float64,ComplexF64)
        un = (Ten(yy=_rand(T1)),
              Ten(xx=_rand(T1),xz=_rand(T1), zx=_rand(T1), zz=_rand(T1)),
              Ten(xx=_rand(T1),xy=_rand(T1),xz=_rand(T1),
                  yx=_rand(T1),yy=_rand(T1),yz=_rand(T1),
                  zx=_rand(T1),zy=_rand(T1),zz=_rand(T1)))
        for u in un
            Au = Array{TensorsLite._my_eltype(u)}(u)
            for op in (+,-,normalize)
                @test op(u) ≈ op(Au)
            end
            @test norm(u) ≈ norm(Au)
        end
        for T2 in (Int64,Float64,ComplexF64)
            vn = (Ten(yy=_rand(T1)),
                  Ten(xx=_rand(T1),xz=_rand(T1), zx=_rand(T1), zz=_rand(T1)),
                  Ten(xx=_rand(T1),xy=_rand(T1),xz=_rand(T1),
                      yx=_rand(T1),yy=_rand(T1),yz=_rand(T1),
                      zx=_rand(T1),zy=_rand(T1),zz=_rand(T1)))
            for u in un
                Au = Array{TensorsLite._my_eltype(u)}(u)
                for v in vn
                    Av = Array{TensorsLite._my_eltype(v)}(v)
                    for op in (+,-,*)
                        @test op(u,v) ≈ op(Au,Av)
                    end
                    @test dot(u,v) ≈ Au*Av
                    @test muladd(2.0,u,v) ≈ (2.0*Au + Av)
                    @test muladd(u,2.0,v) ≈ (2.0*Au + Av)

                    @test muladd(u,v,v) ≈ (Au*Av + Av)
                    @test dotadd(u,v,v) ≈ (Au*Av + Av)
                    @test inner(u,v) ≈ dot(Au,Av)
                end
            end
        end
    end
end

@testset "Tensor x Vec Operations" begin
    for T1 in (Int64,Float64,ComplexF64)
        Tn = (Ten(yy=_rand(T1)),
              Ten(xx=_rand(T1),xz=_rand(T1), zx=_rand(T1), zz=_rand(T1)),
              Ten(xx=_rand(T1),xy=_rand(T1),xz=_rand(T1),
                  yx=_rand(T1),yy=_rand(T1),yz=_rand(T1),
                  zx=_rand(T1),zy=_rand(T1),zz=_rand(T1)))
 
        for T2 in (Int64,Float64,ComplexF64)
            vn = (Vec(y=_rand(T2)), Vec(x=_rand(T2), z=_rand(T2)), Vec(_rand(T2),_rand(T2),_rand(T2)))
            for T in Tn
                AT = Array{TensorsLite._my_eltype(T)}(T)

                @test transpose(T) == transpose(AT)
                @test T' == AT'

                for v in vn
                    Av = Array{TensorsLite._my_eltype(v)}(v)

                    @test T*v ≈ AT*Av
                    @test v*T ≈ transpose(AT)*Av
                    @test muladd(T,v,v) ≈ (AT*Av + Av)
                    @test dotadd(T,v,v) ≈ (AT*Av + Av)
                    @test muladd(v,T,v) ≈ (transpose(AT)*Av + Av)
                    @test dotadd(v,T,v) ≈ (transpose(AT)*Av + Av)
                end
            end
        end
    end
end