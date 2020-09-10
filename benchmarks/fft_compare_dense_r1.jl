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
