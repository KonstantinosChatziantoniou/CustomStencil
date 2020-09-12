using CUDA
using PaddedViews
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

function fft_stencil_3d(data, template, t_steps=1)
    # template is a 3d square array
    #r = (size(template,1)-1)÷2

    r2 = size(template,1)
    r = 32
    while r < r2
        r += 32
    end
    r = r + 1
    dx,dy,dz = size(data)
    padded_data = PaddedView(0, data, (1:dx+r-1, 1:dy+r-1, 1:dz+r-1), (1:dx,1:dy,1:dz))
    padded_template = PaddedView(0, template, (1:dx+r-1, 1:dy+r-1, 1:dz+r-1), (1:r2,1:r2,1:r2))
    ## Initial Upload
    fd = CuArray{CUDA.cuFloatComplex}(padded_data)
    ft = CuArray{CUDA.cuFloatComplex}(padded_template)
    temp = similar(ft)
    CUFFT.fft!(fd)
    CUFFT.fft!(ft)
    #@time m = fd.*ft
    #@time elem_mul_3d!(fd,ft)
    ## Time Loop
    CUFFT.ifft!(fd)
    r2 = r2÷2 + 1
    rn = r2:(r2+dx-1)
    return real(Array(fd))[rn,rn,rn]
end

# function stride_bench(data, rsize)
#     template = rand(Float32, rsize, rsize, rsize)
#     @info "running for $rsize"
#     fft_stencil_3d(data, template, 1)
# end
#
# d = [1<<7 for i = 1:3]
# data = rand(Float32, d...)
# for i in [3 5 9 17 33]
#     global data
#     stride_bench(data, i)
# end
#
# CUDA.cuProfilerStart()
# for i in [3 5 9 17 33]
#     NVTX.@range "template$(i)" begin
#         global data
#         stride_bench(data, i)
#     end
# end


# function locFFTstencil(st_inst::StencilInstance, org_data, t_steps::Integer, multi=1)
#     @show z = size(org_data, 3)
#     template_single = matrix_from_expression(st_inst.stencil_sym, st_inst.max_radius)
#     template_multi = nothing
#     if multi > 1
#         template_multi = DSP.conv(template_single, template_single)
#         for i = 2:multi
#             template_multi = DSP.conv(template_multi, template_single)
#         end
#     end
include("../../module/CustomStencil.jl")
include("../../misc/misc.jl")
include("../../misc/cpu_stencils.jl")
coefs = [0.75;0.5;0.25]
star_stencil = @def_stencil_expression c[0]D[x,y,z] + @sum(i, 1,2, c[i]*(
            D[x+i,y,z] + D[x,y+i,z] + D[x,y,z+i] +
            D[x-i,y,z] + D[x,y-i,z] + D[x,y,z-i]))
st_def = CreateStencilDefinition(star_stencil, coefs)
st_inst1 = NewStencilInstance(st_def, m_step=false)

radius = 4
nx = 5
ny = 5
nz = 5
dx = 1<<(nx)
dy = 1<<(ny)
dz = 1<<(nz)

data = CreateData(dx,dy,dz)
gpu_out = ApplyStencil(st_inst1, data, 1)


t = matrix_from_expression(st_inst1.stencil_sym, st_inst1.max_radius)
a = fft_stencil_3d(data, t, 1)

b = a
b[.*(a .< 0.00001)] .= 0
c = gpu_out[10:20,10:20,16]
d = b[10:20,10:20,18]

using Plots

heatmap(a[:,:,1])
heatmap(gpu_out[:,:,4])


sum(a .- gpu_out)
sum(b .- gpu_out)
max((a.-gpu_out)...)
max((b.-gpu_out)...)

sum(b .< 0.00001)
