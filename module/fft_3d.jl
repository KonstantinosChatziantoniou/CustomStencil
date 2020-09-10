using CUDA
using PaddedViews


function k_index_kernel_fft_3d(data, out, offx, offy, offz)
    dx = (blockIdx().x-1)*blockDim().x + threadIdx().x
    dy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    dz = (blockIdx().z-1)*blockDim().z + threadIdx().z

    sx,sy,sz = size(data)
    sx = sx - offx
    sy = sy - offy
    sz = sz - offz
    if dx > sx || dy > sy || dz > sz
        return nothing
    end

    @inbounds out[dx,dy,dz] = data[dx+offx, dy+offy, dz + offz]

    return nothing
end

function k_zeroing_3d(data, offx, offy, offz)
    sx,sy,sz = size(data)
    dx = (blockIdx().x-1)*blockDim().x + threadIdx().x
    dy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    dz = 1
    for dz = 1:offx
        if dx <= sx && dy <= sy && dz <= offz
            @inbounds data[dx, dy, end-dz+1] = 0
        end
        if dx <= sx && dy <= sz && dz <= offy
            @inbounds data[dx, end-dz+1, dy] = 0
        end
        if dx <= sz && dy <= sy && dz <= offx
            @inbounds data[end-dz+1, dy, dx] = 0
        end
    end
    return nothing
end

function cu_offset_result_3d(data, out, offx, offy, offz)
    dims = size(data)
    tx = 16; ty = 8; tz = 8
    bx = 1 + dims[1]÷tx
    by = 1 + dims[2]÷ty
    bz = 1 + dims[3]÷tz
    @cuda blocks=(bx,by, bz) threads=(tx,ty, tz) k_index_kernel_fft_3d(data,out,offx,offy,offz)
    tx = 32
    ty = 32
    #tz = 2max(offy,offx)
    bx = 1 + max(dims...)÷tx
    by = 1 + max(dims...)÷ty
    bz = 1
    #println("BLOCKS THREADS ",(bx,by,bz) ,(tx,ty,tz) , size(out), typeof(out))
    @cuda blocks=(bx,by) threads=(tx,ty) k_zeroing_3d(out,2offx,2offy,2offz)
end

function k_elem_mul_3d!(A, B)
    tx = (blockIdx().x-1)*blockDim().x + threadIdx().x
    ty = (blockIdx().y-1)*blockDim().y + threadIdx().y
    tz = (blockIdx().z-1)*blockDim().z + threadIdx().z
    sx,sy,sz = size(A)
    if tx > sx || ty > sy || tz > sz
        return nothing
    end
    @inbounds a = A[tx,ty,tz]
    @inbounds b = B[tx,ty,tz]
    @inbounds A[tx,ty,tz] = a*b
    return nothing
end

function elem_mul_3d!(A,B)
    sx,sy,sz = size(A)
    tx = 16; ty = 8; tz = 8

    bx = 1 + sx÷tx
    by = 1 + sy÷ty
    bz = 1 + sz÷tz
    @cuda blocks=(bx,by,bz) threads=(tx,ty,tz) k_elem_mul_3d!(A,B)
    nothing
end

function k_cmplx_to_real_3d!(A)
    dx = (blockIdx().x-1)*blockDim().x + threadIdx().x
    dy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    dz = (blockIdx().z-1)*blockDim().z + threadIdx().z
    sx,sy,sz = size(A)
    if dx > sx || dy > sy || dz > sz
        return nothing
    end
    @inbounds a = A[dx,dy,dz]
    @inbounds A[dx,dy,dz] = a.re
    return nothing
end

function cmplx_to_real_3d!(A)
    sx,sy,sz = size(A)
    tx = 16; ty = 8; tz = 8
    bx = 1 + sx÷tx
    by = 1 + sy÷ty
    bz = 1 + sz÷tz
    @cuda blocks=(bx,by,bz) threads=(tx,ty,tz) k_cmplx_to_real_3d!(A)
    nothing
end



function fft_stencil_3d(data, template, t_steps=1)
    # template is a 3d square array
    r = (size(template,1)-1)÷2
    dx,dy,dz = size(data)
    padded_data = PaddedView(0, data, (1:dx+2r, 1:dy+2r, 1:dz+2r), (1:dx,1:dy,1:dz))
    padded_template = PaddedView(0, template, (1:dx+2r, 1:dy+2r, 1:dz+2r), (1:2r+1,1:2r+1,1:2r+1))
    ## Initial Upload
    fd = CuArray{CUDA.cuFloatComplex}(padded_data)
    ft = CuArray{CUDA.cuFloatComplex}(padded_template)
    temp = similar(ft)
    CUFFT.fft!(fd)
    CUFFT.fft!(ft)
    elem_mul_3d!(fd,ft)
    ## Time Loop
    CUFFT.ifft!(fd)
    cu_offset_result_3d(fd, temp, r,r,r)
    cmplx_to_real_3d!(temp)
    fd,temp = temp,fd
    for t = 2:t_steps
        CUFFT.fft!(fd)
        elem_mul_3d!(fd,ft)
        CUFFT.ifft!(fd)
        cu_offset_result_3d(fd, temp, r,r,r)
        cmplx_to_real_3d!(temp)
        fd,temp = temp,fd
    end
    return view(real(Array(fd)), 1:dx, 1:dy, 1:dz)
end
