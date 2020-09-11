include("../../module/CustomStencil.jl")
include("../../misc/misc.jl")
include("../../misc/cpu_stencils.jl")
## Star Stencil definition with radius = 4
t_steps = 4
if length(ARGS) == 1
    t_steps = parse(Int, ARGS[1])
end
st_insts = []
for i in [1 2 4 8 16 32]
    global st_insts
    coefs = round.([1/j for j = 1:(i+1)],digits=4)
    star_stencil = def_stencil_expression(:(c[0]D[x,y,z] + @sum(i, 1,$i, c[i]*(
                D[x+i,y,z] + D[x,y+i,z] + D[x,y,z+i] +
                D[x-i,y,z] + D[x,y-i,z] + D[x,y,z-i]))))

    st_def = CreateStencilDefinition(star_stencil, coefs)
    bdim = 32
    if i > 8
        bdim = 16
    end
    st_inst = NewStencilInstance(st_def, m_step=false, bdim=bdim)
    push!(st_insts, st_inst)

end


## Warm Up
function warmup(st_insts)
    nx = 6
    ny = 6
    nz = 6
    dx = 1<<(nx)
    dy = 1<<(ny)
    dz = 1<<(nz)

    data = CreateData(dx,dy,dz)

    t_steps = 1

    gpu_out = ApplyFFTstencil(st_insts[1], data, t_steps)
end
warmup(st_insts)

## Benchmark
function bench(st_insts)
    nx = 8
    ny = 8
    nz = 8
    dx = 1<<(nx)
    dy = 1<<(ny)
    dz = 1<<(nz)

    data = CreateData(dx,dy,dz)

    global t_steps

    CUDA.cuProfilerStart()
    for i in st_insts
        ##global data, t_steps
        NVTX.@range "r$(i.max_radius)" begin
            gpu_out = ApplyFFTstencil(i, data, t_steps, t_steps)
        end
    end
end
bench(st_insts)
exit()
