include("../module/CustomStencil.jl")
include("../misc/misc.jl")
include("../misc/cpu_stencils.jl")
## Star Stencil definition with radius = 4
coefs = [1;-0.5;0.25;-0.125;0.05]
star_stencil = @def_stencil_expression c[0]D[x,y,z] + @sum(i, 1, 4, c[i]*(
            D[x+i,y,z] + D[x,y+i,z] + D[x,y,z+i] +
            D[x-i,y,z] + D[x,y-i,z] + D[x,y,z-i]))
st_def = CreateStencilDefinition(star_stencil, coefs)
st_single = NewStencilInstance(st_def, m_step=false)
st_16bdim =  NewStencilInstance(st_def, m_step=false, bdim=16)

coefs = [1;-0.5;0.25;-0.125;0.05;-0.5;0.25;-0.125;0.05]
star_stencil = @def_stencil_expression c[0]D[x,y,z] + @sum(i, 1, 8, c[i]*(
            D[x+i,y,z] + D[x,y+i,z] + D[x,y,z+i] +
            D[x-i,y,z] + D[x,y-i,z] + D[x,y,z-i]))
st_def = CreateStencilDefinition(star_stencil, coefs)
st_inst = NewStencilInstance(st_def, m_step=false)
## Input Data size Definition
radius = 4
nx = 9
ny = 9
nz = 9
dx = 1<<(nx)
dy = 1<<(ny)
dz = 1<<(nz)

data = CreateData(dx,dy,dz)

t_steps = 4
## GPU Implementation
gpu_out = ApplyStencil(st_inst, data, t_steps)
gpu_out = ApplyStencil(st_single, data, t_steps)
gpu_out = ApplyStencil(st_16bdim, data, t_steps)

CUDA.cuProfilerStart()
gpu_out = ApplyStencil(st_inst, data, t_steps)
gpu_out = ApplyStencil(st_single, data, t_steps)
gpu_out = ApplyStencil(st_16bdim, data, t_steps)

exit()
