module StrideArraysCoreExt

using TensorsLite, StrideArraysCore

@inline get_object(a) = StrideArraysCore.object_and_preserve(a)[1]

@inline get_object(a::Array{T,N}) where {T<:Union{TensorsLite.Zeros.Zero,TensorsLite.Zeros.One}, N} = a

@inline StrideArraysCore.object_and_preserve(a::VecArray) = (VecArray(get_object(a.x), get_object(a.y), get_object(a.z)), a)

@inline StrideArraysCore.object_and_preserve(a::AntiSymTenArray) = (AntiSymTenArray(get_object(a.yx), get_object(a.zx), get_object(a.zy)), a)

@inline StrideArraysCore.object_and_preserve(a::SymTenArray) = (SymTenArray(get_object(a.xx), get_object(a.yx), get_object(a.zx),
                                                                                              get_object(a.yy), get_object(a.yz),
                                                                                                                get_object(a.zz)),
                                                                a)

end
