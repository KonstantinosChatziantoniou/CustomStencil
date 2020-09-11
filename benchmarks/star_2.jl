include("../module/CustomStencil.jl")
include("../misc/misc.jl")
include("../misc/cpu_stencils.jl")
## Star Stencil definition with radius = 2
coefs = [1;0.5;0.25]
star_stencil = @def_stencil_expression c[0]D[x,y,z] + @sum(i, 1, 2, c[i]*(
            D[x+i,y,z] + D[x,y+i,z] + D[x,y,z+i] +
            D[x-i,y,z] + D[x,y-i,z] + D[x,y,z-i]))
st_def = CreateStencilDefinition(star_stencil, coefs)
st_inst = NewStencilInstance(st_def, m_step=2)  #don't combine time steps
st_single = NewStencilInstance(st_def, m_step=false)

## Input Data size Definition
radius = 2
nx = 8
ny = 8
nz = 8
dx = 1<<(nx)
dy = 1<<(ny)
dz = 1<<(nz)

data = CreateData(dx,dy,dz)

t_steps = 10
## GPU Implementation
gpu_out = ApplyStencil(st_inst, data, t_steps)
gpu_out = ApplyStencil(st_single, data, t_steps)

CUDA.cuProfilerStart()
gpu_out = ApplyStencil(st_inst, data, t_steps)
gpu_out = ApplyStencil(st_single, data, t_steps)

exit()
## CPU result
cpu_out = cpu_star_stencil(data, radius, coefs, t_steps)

## Check
mean_er = sum(abs.(cpu_out .- gpu_out))./(reduce(*, size(gpu_out)))
heatmap(gpu_out[:,:,Int(round(dz/2))+shift]-cpu_out[:,:,Int(round(dz/2))+shift])



@assert(mean_er/max(cpu_out...) < 0.000001)
## Visualize Error
using Plots


shift = 0
Int(round(dz/2))+shift
heatmap(gpu_out[:,:,Int(round(dz/2))+shift]-cpu_out[:,:,Int(round(dz/2))+shift])
heatmap(cpu_out[:,:,Int(round(dz/2))+shift])
heatmap(gpu_out[:,:,Int(round(dz/2))+shift])


function init_coeff_arr(cf)
    ex = :(cfarr = SA_F32[])
    for i in cf
        push!(ex.args[2].args , i)
    end
    return ex
end

ex = :(cfarr = SA_F32[1])

using StaticArrays

a = SA_F32[1,2,3]


function ch_ker(arr)
    a = SA_F32[1,2,3,0.5]
    for i = 1:4
        arr[threadIdx().x] += a[i]
    end
    return nothing
end

ar = CUDA.ones(Float32, 32)
@cuda threads=32 ch_ker(ar)
