using CUDA
using PaddedViews


function k_index_kernel_fft_2d(data, out,offx, offy)
    dx = (blockIdx().x-1)*blockDim().x + threadIdx().x
    dy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    #dz = (blockIdx().z-1)*blockDim().z + threadIdx().z

    #sx,sy,sz = size(data)
    # if dx > sx || dy > sy || dz > sz
    #     return nothing
    # end
    sx,sy = size(data)
    sx = sx - offx
    sy = sy - offy
    if dx > sx || dy > sy
        return nothing
    end

    out[dx,dy] = data[dx+offx, dy+offy]

    return nothing
end

function k_zeroing_2d(data, offx, offy)
    sx,sy = size(data)
    dx = (blockIdx().x-1)*blockDim().x + threadIdx().x
    dy = threadIdx().y
    if dx <= sx && dy <= offy
        @inbounds data[dx, end-dy+1] = 0
    end
    if dx <= sy && dy <= offx
        @inbounds data[end-dy+1, dx] = 0
    end
    return nothing
end

function cu_offset_result_2d(data, out, offx, offy)
    dims = size(data)
    tx = 32; ty =32
    bx = 1 + dims[1]÷tx
    by = 1 + dims[2]÷ty

    @cuda blocks=(bx,by) threads=(tx,ty) k_index_kernel_fft_2d(data,out,offx,offy)

    @show ty = 2*max(offy,offx)
    @show bx = 1 + max(dims...)÷tx
    @show by = 1
    @cuda blocks=(bx,by) threads=(tx,ty) k_zeroing_2d(out,2offx,2offy)
end

function k_elem_mul_2d!(A, B)
    tx = (blockIdx().x-1)*blockDim().x + threadIdx().x
    ty = (blockIdx().y-1)*blockDim().y + threadIdx().y
    sx,sy = size(A)
    if tx > sx || ty > sy
        return nothing
    end
    @inbounds a = A[tx,ty]
    @inbounds b = B[tx,ty]
    @inbounds A[tx,ty] = a*b
    return nothing
end

function elem_mul_2d!(A,B)
    sx,sy = size(A)
    tx = 32
    ty = 32
    bx = 1 + sx÷tx
    by = 1 + sy÷ty
    @cuda blocks=(bx,by) threads=(tx,ty) k_elem_mul_2d!(A,B)
    nothing
end

function k_cmplx_to_real_2d!(A)
    dx = (blockIdx().x-1)*blockDim().x + threadIdx().x
    dy = (blockIdx().y-1)*blockDim().y + threadIdx().y
    sx,sy = size(A)
    if dx > sx || dy > sy
        return nothing
    end
    a = A[dx,dy]
    A[dx,dy] = a.re
    return nothing
end

function cmplx_to_real_2d!(A)
    sx,sy = size(A)
    tx = 32
    ty = 32
    bx = 1 + sx÷tx
    by = 1 + sy÷ty
    @cuda blocks=(bx,by) threads=(tx,ty) k_cmplx_to_real_2d!(A)
    nothing
end



function fft_stencil_2d(data, template, t_steps=1)
    # template is a 2d square array
    r2 = size(template,1)
    r = 32
    while r <= r2
        r += 32
    end
    r = r + 1
    print(template)
    print((size(data)))
    dx,dy = size(data)
    padded_data = PaddedView(0, data, (1:dx+r-1, 1:dy+r-1), (1:dx,1:dy))
    padded_template = PaddedView(0, template, (1:dx+r-1, 1:dy+r-1,), (1:r2,1:r2))
    ## Initial Upload
    fd = CuArray{CUDA.cuFloatComplex}(padded_data)
    ft = CuArray{CUDA.cuFloatComplex}(padded_template)
    temp = similar(ft)
    CUFFT.fft!(fd)
    CUFFT.fft!(ft)
    elem_mul_2d!(fd,ft)
    ## Time Loop
    CUFFT.ifft!(fd)
    r2 = r2÷2 + 1
    rn = r2:(r2+dx-1)
    return real(Array(fd))[rn,rn]
end
