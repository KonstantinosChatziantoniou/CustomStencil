include("../module/CustomStencil.jl")
include("../misc/misc.jl")
include("../misc/cpu_stencils.jl")
## Star Stencil definition with radius = 4
coefs = [1;-0.5;0.25;-0.125;0.05]
star_stencil = @def_stencil_expression c[0]D[x,y,z] + @sum(i, 1, 4, c[i]*(
            D[x+i,y,z] + D[x,y+i,z] + D[x,y,z+i] +
            D[x-i,y,z] + D[x,y-i,z] + D[x,y,z-i]))
st_def = CreateStencilDefinition(star_stencil, coefs)
st_inst = NewStencilInstance(st_def, m_step=false)
st_inst2 = NewStencilInstance(st_def, m_step=false)
## Input Data size Definition
radius = 4
nx = 8
ny = 8
nz = 8
dx = 1<<(nx)
dy = 1<<(ny)
dz = 1<<(nz)

data = CreateData(dx,dy,dz)

t_steps = 4
## GPU Implementation
gpu_out = ApplyStencil(st_inst, data, t_steps)

CUDA.cuProfilerStart()
t_steps = 4
data = data
gpu_out = ApplyStencil(st_inst, data, t_steps)

t_steps = 8
data = data[:,:,1:(dz÷2)]
gpu_out = ApplyStencil(st_inst2, data, t_steps)
exit()
