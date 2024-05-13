using TensorsLite
using Zeros
using Test
using LinearAlgebra

const ze = Zero()

# This helps comparing arrays of Union{Zero,Number}
(::Type{<:Union{Zero,T}})(x::Number) where T<: Number = x == zero(x) ? Zeros.Zero() : T(x)

@testset "_muladd definitions" begin
    for x in (One(),Zero(),rand())
        for y in (One(),Zero(),rand())
            for z in (One(),Zero(),rand())
                if (all(t->typeof(t)===Float64,(x,y,z)))
                    @test TensorsLite._muladd(x,y,z) === muladd(x,y,z)
                else
                    @test TensorsLite._muladd(x,y,z) ≈ x*y + z
                end
            end
        end
    end
    @test TensorsLite._muladd(Zero(),1.0im,2.0) === 2.0 + 0.0im
    @test TensorsLite._muladd(1.0im,Zero(),2.0) === 2.0 + 0.0im
end

@testset "Vec Constructors" begin

    @test Vec(x=1).x === 1
    @test Vec(y=1.0).y === 1.0
    @test typeof(Vec(z=1.0)) === Vec1Dz{Float64}

    @test eltype(Vec(x=1.0,y=2)) === Union{Zero,Float64}

    @test eltype(Vec(x=1,y=3,z=4.0im)) === ComplexF64

    @test eltype(Vec(x=One())) === Union{Zero,One}

end

@testset "Tensor Constructors" begin
    @test Ten(xx=1, xy=2, xz=3,
              yx=4, yy=5, yz=6,
              zx=7, zy=8, zz=9.0) == [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]

    @test eltype(Ten(xx=1, xy=2, xz=3,
                     yx=4, yy=5, yz=6,
                     zx=7, zy=8, zz=9.0)) === Float64

    @test eltype(Ten(xx=1.0)) === Union{Zero,Float64}

    @test eltype(Ten(xx=One())) === Union{Zero,One}

    @test Vec(x=Vec(x=1.0,y=2.0),y=Vec(x=3.0,y=4.0)).z === Vec()
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

_rand(T) = rand(T)
_rand(T::Type{Int64}) = rand((1,2,3,4,5,6,7,8,9,10))
@testset "Vector Operations" begin

    @test Vec(1,2,3) + [1,2,3] == [2,4,6]
    @test [1,2,3] + Vec(1,2,3) == [2,4,6]
    @test Vec(3,2,1) - [1,2,3] == [2,0,-2]
    @test [1,2,3] - Vec(3,2,1) == [-2,0,2]

    @test convert(Vec3D{Float64}, 𝐢) === Vec(1.0,0.0,0.0)
    @test convert(Vec3D{Float64}, 𝐤) === Vec(0.0,0.0,1.0)
    @test convert(Vec2Dxy{Float64}, 𝐢) === Vec(x=1.0,y=0.0)
    @test convert(Vec2Dxz{Float64}, 𝐤) === Vec(x=0.0,z=1.0)

    @test zero(1.0𝐢) === Vec(x=0.0)
    @test zero(1𝐣) === Vec(y=0)
    @test zero(Vec(y=1.0,z=im)) === Vec(y=zero(1.0im),z=zero(1.0im))

    @test Vec(1,2,3)//4 === Vec(1//4,1//2,3//4)

    @test +(Vec(1,1,1),Vec(1,1,1),Vec(1,1,1),Vec(1,1,1)) === Vec(4,4,4)

    @test norm(-2.0𝐢) === 2.0
    @test norm(-2.0𝐤) === 2.0
    @test norm(Vec()) === Zero()

    @test normalize(Vec()) === Vec()

    @test Vec(x=1.0) ≈ Vec(x=1.0,y=eps())
    @test isapprox(Vec(x=1.0), Vec(x=1.0,y=eps()); rtol=sqrt(eps()))

    for T1 in (Int64,Float64,ComplexF64)
        un = (Vec(y=_rand(T1)), Vec(x=_rand(T1), z=_rand(T1)), Vec(_rand(T1),_rand(T1),_rand(T1)))
        for u in un
            Au = Array{nonzero_eltype(u)}(u)
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
                Au = Array{nonzero_eltype(u)}(u)
                for v in vn
                    Av = Array{nonzero_eltype(v)}(v)
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

    @test Ten(xx=1.0) ≈ Ten(xx=1.0,xy=eps())
    @test isapprox(Ten(xx=1.0), Ten(xx=1.0,xy=eps()); rtol=sqrt(eps()))

    for T1 in (Int64,Float64,ComplexF64)
        un = (Ten(yy=_rand(T1)),
              Ten(xx=_rand(T1),xz=_rand(T1), zx=_rand(T1), zz=_rand(T1)),
              Ten(xx=_rand(T1),xy=_rand(T1),xz=_rand(T1),
                  yx=_rand(T1),yy=_rand(T1),yz=_rand(T1),
                  zx=_rand(T1),zy=_rand(T1),zz=_rand(T1)))
        for u in un
            Au = Array{nonzero_eltype(u)}(u)
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
                Au = Array{nonzero_eltype(u)}(u)
                for v in vn
                    Av = Array{nonzero_eltype(v)}(v)
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
                AT = Array{nonzero_eltype(T)}(T)

                @test transpose(T) == transpose(AT)
                @test T' == AT'

                for v in vn
                    Av = Array{nonzero_eltype(v)}(v)

                    @test T*v ≈ AT*Av
                    @test v*T ≈ transpose(AT)*Av
                    @test muladd(T,v,v) ≈ (AT*Av + Av)
                    @test dotadd(T,v,v) ≈ (AT*Av + Av)
                    @test muladd(v,T,v) ≈ (transpose(AT)*Av + Av)
                    @test dotadd(v,T,v) ≈ (transpose(AT)*Av + Av)
                    @test dot(v,T,v) ≈ dot(conj(Av), AT*Av)
                end
            end
        end
    end
end

@testset "SymTen" begin
    @test typeof(SymTen(1,2,3,4,5,6)) === SymTen3D{Int}
    @test typeof(SymTen(xx=1.0,yx=2,yy=3)) === SymTen2Dxy{Float64}
    @test typeof(SymTen(xx=1.0,zx=2,zz=3)) === SymTen2Dxz{Float64}
    @test typeof(SymTen(yy=1.0,zy=2,zz=3)) === SymTen2Dyz{Float64}

    @test -SymTen(1,2,3,4,5,6) === SymTen(-1,-2,-3,-4,-5,-6)

    @test SymTen(1,2,3,4,5,6) == [1 2 3;
                                  2 4 5;
                                  3 5 6]

    @test SymTen(6,5,4,3,2,0) == SymTen(6.0,5.0,4.0,3.0,2.0,0.0)

    @test (SymTen(xx=1) + SymTen(xx=2.0, yy=5.0)) === SymTen(xx=3.0,yy=5.0)
    @test (SymTen(xx=1) - SymTen(xx=2.0, yy=5.0)) === SymTen(xx=-1.0,yy=-5.0)
    @test muladd(3.0,SymTen(xx=4.0),SymTen(xx=5.0)) === SymTen(xx=17.0)
    @test muladd(One(),SymTen(1,2,3,4,5,6),SymTen()) === SymTen(1,2,3,4,5,6)

    let xx=rand(ComplexF64),yx = rand(ComplexF64),zx=rand(ComplexF64),yy=rand(ComplexF64),zy=rand(ComplexF64),zz=rand(ComplexF64)
        S = SymTen(xx,yx,zx,yy,zy,zz)
        @test S.x === Vec(xx,yx,zx)
        @test S.y === Vec(yx,yy,zy)
        @test S.z === Vec(zx,zy,zz)
        @test transpose(S) === S
        @test S' === SymTen(conj(xx),conj(yx),conj(zx),conj(yy),conj(zy),conj(zz))
        @test S.xy === S.yx
        @test S.zy === S.yz
        @test S.zx === S.xz
    end

    @test convert(SymTen2Dxy{Float64},SymTen(xx=1)) === SymTen(xx=1.0,yx=0.0,yy=0.0)
    @test convert(Ten2Dxz{Float64},SymTen(zx=1)) === Ten(xx=0.0,zx=1.0,xz =1.0,zz=0.0)

end

@testset "AntiSymTen" begin
    @test typeof(AntiSymTen(yx=1.0,zx=2,zy=3)) === AntiSymTen3D{Float64}
    @test typeof(AntiSymTen(yx=1.0)) === AntiSymTen2Dxy{Float64}
    @test typeof(AntiSymTen(zy=2)) === AntiSymTen2Dyz{Int}
    @test typeof(AntiSymTen(zx=2.0)) === AntiSymTen2Dxz{Float64}

    @test -AntiSymTen(1,2,3) === AntiSymTen(-1,-2,-3)

    @test AntiSymTen(1,2,3) == [0 -1 -2;
                                1  0 -3;
                                2  3  0]

    @test AntiSymTen(3,2,1) == AntiSymTen(3.0,2.0,1.0)

    @test (AntiSymTen(yx=1) + AntiSymTen(yx=2.0, zx=5.0)) === AntiSymTen(yx=3.0,zx=5.0)
    @test (AntiSymTen(yx=1) - AntiSymTen(yx=2.0, zx=5.0)) === AntiSymTen(yx=-1.0,zx=-5.0)
    @test muladd(3.0,AntiSymTen(yx=4.0),AntiSymTen(yx=5.0)) === AntiSymTen(yx=17.0)
    @test muladd(One(),AntiSymTen(1,2,3),AntiSymTen()) === AntiSymTen(1,2,3)

    let yx=rand(ComplexF64),zx=rand(ComplexF64),zy=rand(ComplexF64)
        W = AntiSymTen(yx,zx,zy)
        @test W.x === Vec(y=yx,z=zx)
        @test W.y === Vec(x=-yx,z=zy)
        @test W.z === Vec(x=-zx,y=-zy)
        @test transpose(W) === -W
        @test W' === AntiSymTen(-conj(yx),-conj(zx),-conj(zy))
    end

    @test convert(AntiSymTen3D{Float64},AntiSymTen(yx=1)) === AntiSymTen(yx=1.0,zx=0.0,zy=0.0)
    @test convert(Ten2Dxz{Float64},AntiSymTen(zx=1)) === Ten(xx=0.0,zx=1.0,xz =-1.0,zz=0.0)

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

    let a = VecArray(x=rand(Float32,2),y=rand(Float32,2))
        @test typeof(similar(a,Vec2Dyz{Int},(4,4,4))) === Vec2DyzArray{Int,3}
    end

    let a = TenArray(xx=rand(2),xz=rand(2),zx=rand(2),zz=rand(2))
        typeof(similar(a,Ten3D{Float16},(1,1,1))) === Ten3DArray{Float16,3}
    end

    let a1=[1],a2=[2],a3=[3],a4=[4],a5=[5],a6=[6],a7=[7],a8=[8],a9=[9],T=TenArray(a1,a2,a3,a4,a5,a6,a7,a8,a9)
        @test T.xx === a1
        @test T.yx === a2
        @test T.zx === a3
        @test T.xy === a4
        @test T.yy === a5
        @test T.zy === a6
        @test T.xz === a7
        @test T.yz === a8
        @test T.zz === a9
    end

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

@testset "TenArray Broadcasting" begin
    let ux=TenArray(xx=rand(3)), uxy=TenArray(xx=rand(Float32,3),yx=rand(Float32,3),xy=rand(Float32,3),yy=rand(Float32,3))
        @test (ux .+ uxy) == TenArray(xx=ux.x.x .+ uxy.x.x, xy=uxy.y.x, yx=uxy.x.y, yy=uxy.y.y)
        @test typeof(ux .+ uxy) === Ten2DxyArray{Float64,1}
        @test ux .+ (𝐢 ⊗ 𝐢) == TenArray(xx=ux.x.x .+ 1)
        @test (ux .= Ten()) == TenArray(xx=zeros(3))
    end
end
