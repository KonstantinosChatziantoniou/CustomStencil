include("../module/CustomStencil.jl")
include("../misc/misc.jl")
include("../misc/cpu_stencils.jl")
## Star Stencil definition with radius = 4
coefs = [0.75;0.5]
star_stencil = @def_stencil_expression c[0]D[x,y,z] + @sum(i, 1,1, c[i]*(
            D[x+i,y,z] + D[x,y+i,z] +
            D[x-i,y,z] + D[x,y-i,z]))
st_def = CreateStencilDefinition(star_stencil, coefs)
st_inst1 = NewStencilInstance(st_def, m_step=false)
st_inst2 = NewStencilInstance(st_def, m_step=2)
st_inst3 = NewStencilInstance(st_def, m_step=3)
st_inst4 = NewStencilInstance(st_def, m_step=4)

## Input Data size Definition
radius = 4
nx = 8
ny = 8
nz = 0
dx = 1<<(nx)
dy = 1<<(ny)
dz = 1<<(nz)

data = CreateData(dx,dy,dz)

t_steps = 24
## Compile Functions
gpu_out = ApplyStencil(st_inst1, data, t_steps)
gpu_out = ApplyStencil(st_inst2, data, t_steps)
gpu_out = ApplyStencil(st_inst3, data, t_steps)
gpu_out = ApplyStencil(st_inst4, data, t_steps)

CUDA.cuProfilerStart()
gpu_out = ApplyStencil(st_inst1, data, t_steps)
gpu_out = ApplyStencil(st_inst2, data, t_steps)
gpu_out = ApplyStencil(st_inst3, data, t_steps)
gpu_out = ApplyStencil(st_inst4, data, t_steps)
exit()
