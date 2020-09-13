include("../module/CustomStencil.jl")
include("../misc/misc.jl")
include("../misc/cpu_stencils.jl")
## Star Stencil definition with radius = 4
coefs = [0.75;0.5]
star_stencil = @def_stencil_expression c[0]D[x,y,z] + @sum(i, 1,1, c[i]*(
            D[x+i,y,z] + D[x,y+i,z] + D[x,y,z+i] +
            D[x-i,y,z] + D[x,y-i,z] + D[x,y,z-i]))
st_def = CreateStencilDefinition(star_stencil, coefs)
st_inst = NewStencilInstance(st_def, m_step=false)

dsize = parse(Int, ARGS[1])
t_steps = parse(Int, ARGS[2])
t_group = parse(Int, ARGS[3])
ngpus = parse(Int, ARGS[4])
nx = dsize
ny = dsize
nz = dsize
dx = 1<<(nx)
dy = 1<<(ny)
dz = 1<<(nz)

data = CreateData(dx,dy,dz);


g1 = ApplyStencil(st_inst, data, t_steps)
g2 = ApplyMultiGPU(ngpus , st_inst, t_steps, data, t_group=t_group)


println("AFTER WARMUP")

CUDA.cuProfilerStart()
NVTX.@range "single" begin
g1 = ApplyStencil(st_inst, data, t_steps)end
NVTX.@range "multi" begin
g2 = ApplyMultiGPU(ngpus , st_inst, t_steps, data, t_group=t_group) end
