include("../../module/CustomStencil.jl")
include("../../misc/misc.jl")
include("../../misc/cpu_stencils.jl")
## Star Stencil definition with radius = 2
coefs = Float64.([0.25; 0.125; 0.06])
dense_stencil = @def_stencil_expression(
            @sum(i,-2,2,
                @sum(j,-2,2,
                    @sum(k,-2,2, c[max(abs(i),abs(j),abs(k))]*D[x+i,y+j,z+k]))))
st_def = CreateStencilDefinition(dense_stencil, coefs)
st_inst = NewStencilInstance(st_def)
## Input Data size Definition
radius = 2
nx = 5
ny = 5
nz = 5
dx = 1<<(nx)
dy = 1<<(ny)
dz = 1<<(nz)

data = CreateData(dx,dy,dz)

t_steps = 5
## GPU Implementation
gpu_out = ApplyStencil(st_inst, data, t_steps)

## CPU result
cpu_coefs = fill(0.06, 5,5,5)
cpu_coefs[2:4, 2:4, 2:4] .= 0.125
cpu_coefs[3,3,3] = 0.25
cpu_out = cpu_dense_stencil(data, radius, cpu_coefs, t_steps)

## Check
mean_er = sum(abs.(cpu_out .- gpu_out))./(reduce(*, size(gpu_out)))
@assert(mean_er < 0.001)

## Visualize Error
using Plots


shift = 0
Int(round(dz/2))+shift
heatmap(gpu_out[:,:,Int(round(dz/2))+shift]-cpu_out[:,:,Int(round(dz/2))+shift])
heatmap(cpu_out[:,:,Int(round(dz/2))+shift])
heatmap(gpu_out[:,:,Int(round(dz/2))+shift])
