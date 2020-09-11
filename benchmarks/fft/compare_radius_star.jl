include("../../module/CustomStencil.jl")
include("../../misc/misc.jl")
include("../../misc/cpu_stencils.jl")
## Star Stencil definition with radius = 4
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

for i in st_insts
    #global data, t_steps
    gpu_out = ApplyStencil(i, data, t_steps)
end
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

    t_steps = 1

    CUDA.cuProfilerStart()
    for i in st_insts
        #global data, t_steps

        NVTX.@range "r$(i.max_radius)" begin
            gpu_out = ApplyStencil(i, data, t_steps)
        end
    end
end
bench(st_insts)
exit()
