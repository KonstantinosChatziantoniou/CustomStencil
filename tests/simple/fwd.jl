include("../../module/CustomStencil.jl")
include("../../misc/misc.jl")
include("../../misc/cpu_stencils.jl")
## Star Stencil definition with radius = 2
# first deriv  4 acccuracy
coefs = [0;2/3;-1/12]
fwd = @def_stencil_expression c[0]D[x,y,z] +
            @sum(i, 1, 2,  c[i]*(D[x+i,y,z] + D[x,y+i,z] + D[x,y,z+i])) +
            @sum(i, 1, 2, -c[i]*(D[x-i,y,z] + D[x,y-i,z] + D[x,y,z-i]))
st_def = CreateStencilDefinition(fwd, coefs)
st_inst = NewStencilInstance(st_def, m_step=false, prev_time_coeff=1)  #don't combine time steps


## Input Data size Definition
radius = 2
nx = 5
ny = 5
nz = 5
dx = 1<<(nx)
dy = 1<<(ny)
dz = 1<<(nz)

data = CreateData(dx,dy,dz)

t_steps = 20
## GPU Implementation
gpu_out = ApplyStencil(st_inst, data, t_steps)

## CPU result
cpu_out = cpu_star_stencil(data, radius, coefs, t_steps)

## Check
mean_er = sum(abs.(cpu_out .- gpu_out))./(reduce(*, size(gpu_out)))
@assert(mean_er < 0.001)

## Visualize Error
## Visualize Error
using Plots


shift = 0
Int(round(dz/2))+shift
heatmap(gpu_out[:,:,Int(round(dz/2))+shift]-cpu_out[:,:,Int(round(dz/2))+shift])
heatmap(cpu_out[:,:,Int(round(dz/2))+shift])
heatmap(gpu_out[:,:,Int(round(dz/2))+shift])
