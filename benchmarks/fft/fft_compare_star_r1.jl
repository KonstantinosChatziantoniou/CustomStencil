include("../../module/CustomStencil.jl")
include("../../misc/misc.jl")
include("../../misc/cpu_stencils.jl")
## Star Stencil definition with radius = 4
coefs = [0.75;0.5]
star_stencil = @def_stencil_expression c[0]D[x,y,z] + @sum(i, 1,1, c[i]*(
            D[x+i,y,z] + D[x,y+i,z] + D[x,y,z+i] +
            D[x-i,y,z] + D[x,y-i,z] + D[x,y,z-i]))
st_def = CreateStencilDefinition(star_stencil, coefs)
st_inst1 = NewStencilInstance(st_def, m_step=false)


## Input Data size Definition
radius = 4
nx = 8
ny = 8
nz = 8
dx = 1<<(nx)
dy = 1<<(ny)
dz = 1<<(nz)
data = CreateData(dx,dy,dz)

t_steps = 16
## Compile Functions

gpu_out = ApplyFFTstencil(st_inst1, data, t_steps)
CUDA.cuProfilerStart()
NVTX.@range "r1" begin
    gpu_out = ApplyFFTstencil(st_inst1, data, t_steps,1) end
NVTX.@range "r2" begin
    gpu_out = ApplyFFTstencil(st_inst1, data, t_steps,2) end
NVTX.@range "r3" begin
    gpu_out = ApplyFFTstencil(st_inst1, data, t_steps,4) end
NVTX.@range "r4" begin
    gpu_out = ApplyFFTstencil(st_inst1, data, t_steps,8) end
NVTX.@range "r5" begin
    gpu_out = ApplyFFTstencil(st_inst1, data, t_steps,16) end
exit()
