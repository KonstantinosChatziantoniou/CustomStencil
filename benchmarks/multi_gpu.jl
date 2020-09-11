if !isdefined(Main, :CustomStencil)
    include("../module/CustomStencil.jl")
    using Main.CustomStencil
end
include("../module/CustomStencil.jl")
include("../misc/misc.jl")
include("../misc/cpu_stencils.jl")

## Stencil Definition
coefs = Float64.([1;0.5;0.25])
star_stencil = @def_stencil_expression c[0]D[x,y,z] + @sum(i, 1, 2, c[i]*(
            D[x+i,y,z] + D[x,y+i,z] + D[x,y,z+i] +
            D[x-i,y,z] + D[x,y-i,z] + D[x,y,z-i]))
st_def = CreateStencilDefinition(star_stencil, coefs)
st_inst = NewStencilInstance(st_def, m_step=false)

## Input Data size Definition
radius = 2
nx = 6
ny = 6
nz = 6
dx = 1<<(nx)
dy = 1<<(ny)
dz = 1<<(nz)


t_steps = 4
## GPU Implementation
data = CreateData(dx,dy,dz)
gpu_out = ApplyStencil(st_inst, data, t_steps)

CUDA.cuProfilerStart()
d2 = ApplyMultiGPU(2, st_inst, t_steps, data)
