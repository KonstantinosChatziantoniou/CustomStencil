include("../../module/CustomStencil.jl")
include("../../misc/misc.jl")
include("../../misc/cpu_stencils.jl")
## Star Stencil definition with radius = 2
coefs = fill(0.06, 5,5,5)
coefs[2:4, 2:4, 2:4] .= 0.125
coefs[3,3,3] = 0.25
st_def = CreateStencilDefinition(coefs)
st_inst = NewStencilInstance(st_def, m_step=false)  #don't combine time steps


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
cpu_out = cpu_dense_stencil(data, radius, coefs, t_steps)

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
