include("../module/CustomStencil.jl")
include("../misc/misc.jl")
include("../misc/cpu_stencils.jl")
## Star Stencil definition with radius = 4
coefs = [0.5;0.25]
dense_stencil = @def_stencil_expression(
            @sum(i,-1,1,
                @sum(j,-1,1,
                    @sum(k,-1,1, c[max(abs(i),abs(j),abs(k))]*D[x+i,y+j,z+k]))))

st_def = CreateStencilDefinition(dense_stencil, coefs)
st_inst1 = NewStencilInstance(st_def, m_step=5, bdim=16)
st_inst2 = NewStencilInstance(st_def, m_step=6, bdim=16)
st_inst3 = NewStencilInstance(st_def, m_step=7, bdim=16)
#st_inst4 = NewStencilInstance(st_def, m_step=8, bdim=16)

## Input Data size Definition
radius = 4
nx = 8
ny = 8
nz = 8
dx = 1<<(nx)
dy = 1<<(ny)
dz = 1<<(nz)

data = CreateData(dx,dy,dz)

t_steps = 56
## Compile Functions
gpu_out = ApplyStencil(st_inst1, data, t_steps)
gpu_out = ApplyStencil(st_inst2, data, t_steps)
gpu_out = ApplyStencil(st_inst3, data, t_steps)
#gpu_out = ApplyStencil(st_inst4, data, t_steps)

CUDA.cuProfilerStart()
gpu_out = ApplyStencil(st_inst1, data, t_steps)
gpu_out = ApplyStencil(st_inst2, data, t_steps)
gpu_out = ApplyStencil(st_inst3, data, t_steps)
#gpu_out = ApplyStencil(st_inst4, data, t_steps)

exit()
