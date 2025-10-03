module StrideArraysCoreExt

using TensorsLite, StrideArraysCore, Zeros

@inline get_object(a) = StrideArraysCore.object_and_preserve(a)[1]

@inline get_object(a::Array{T,N}) where {T<:Union{Zeros.Zero,Zeros.One}, N} = a

@inline StrideArraysCore.object_and_preserve(a::TensorArray) = (TensorArray(get_object(a.x), get_object(a.y), get_object(a.z)), a)

@inline StrideArraysCore.object_and_preserve(a::AntiSymTenArray) = (AntiSymTenArray(get_object(a.xy), get_object(a.xz), get_object(a.yz)), a)

@inline StrideArraysCore.object_and_preserve(a::SymTenArray) = (SymTenArray(get_object(a.xx), get_object(a.xy), get_object(a.xz),
                                                                                              get_object(a.yy), get_object(a.zy),
                                                                                                                get_object(a.zz)),
                                                                a)

end
