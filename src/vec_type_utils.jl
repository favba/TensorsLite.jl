@inline _x_type(::Type{<:Vec{T,N,Tx,Ty,Tz}}) where {T,N,Tx,Ty,Tz} = Tx
@inline _y_type(::Type{<:Vec{T,N,Tx,Ty,Tz}}) where {T,N,Tx,Ty,Tz} = Ty
@inline _z_type(::Type{<:Vec{T,N,Tx,Ty,Tz}}) where {T,N,Tx,Ty,Tz} = Tz

@inline function _vec_type(Tx::Type,Ty::Type,Tz::Type)
    Tf = promote_type(Tx,Ty,Tz)
    Txf = _my_promote_type(Tf,Tx)
    Tyf = _my_promote_type(Tf,Ty)
    Tzf = _my_promote_type(Tf,Tz)
    Tff = _final_type(Txf,Tyf,Tzf)
    return Vec{Tff,1,Txf,Tyf,Tzf}
end

@inline function _ten_type(Tx::Type,Ty::Type,Tz::Type)
    _Tx = _my_eltype(Tx) 
    _Ty = _my_eltype(Ty) 
    _Tz = _my_eltype(Tz) 
    Tf = promote_type(_Tx,_Ty,_Tz)
    VTx = _vec_type(_my_promote_type(Tf,_x_type(Tx)), _my_promote_type(Tf,_y_type(Tx)), _my_promote_type(Tf,_z_type(Tx)))
    VTy = _vec_type(_my_promote_type(Tf,_x_type(Ty)), _my_promote_type(Tf,_y_type(Ty)), _my_promote_type(Tf,_z_type(Ty)))
    VTz = _vec_type(_my_promote_type(Tf,_x_type(Tz)), _my_promote_type(Tf,_y_type(Tz)), _my_promote_type(Tf,_z_type(Tz)))
    return Vec{Union{eltype(VTx),eltype(VTy),eltype(VTz)},2,VTx,VTy,VTz}
end
