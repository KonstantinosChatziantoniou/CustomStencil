include("../../module/CustomStencil.jl")
include("../../misc/misc.jl")
include("../../misc/cpu_stencils.jl")
## Star Stencil definition with radius = 4
st_insts = []
for i in [1 2 4 8 16]
    global st_insts
    coefs = round.([1/j for j = 1:(i+1)],digits=4)
    star_stencil = def_stencil_expression(:(c[0]D[x,y,z] + @sum(i, 1,$i, c[i]*(
                D[x+i,y,z] + D[x,y+i,z]+
                D[x-i,y,z] + D[x,y-i,z]))))

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
    nz = 0
    dx = 1<<(nx)
    dy = 1<<(ny)
    dz = 1<<(nz)

    data = CreateData(dx,dy,dz)

    t_steps = 1

    gpu_out = ApplyFFTstencil(st_insts[1], data, t_steps)
        gpu_out = ApplyStencil(st_insts[1], data, 16)
end
warmup(st_insts)

## Benchmark
function bench(st_insts)
    nx = 12
    ny = 12
    nz = 0
    dx = 1<<(nx)
    dy = 1<<(ny)
    dz = 1<<(nz)

    data = CreateData(dx,dy,dz)

    t_steps = 1

    CUDA.cuProfilerStart()
    for i in st_insts
        ##global data, t_steps
        r = i.max_radius
        ms1 = 64÷r
        ms2 = 32÷r
        ms3 = 48÷r
        NVTX.@range "r$(i.max_radius) 3" begin
            gpu_out = ApplyFFTstencil(i, data, t_steps, ms3)
        end
        NVTX.@range "r$(i.max_radius) 0" begin
            gpu_out = ApplyFFTstencil(i, data, t_steps)
        end

        NVTX.@range "r$(i.max_radius) 1" begin
            gpu_out = ApplyFFTstencil(i, data, t_steps, ms1)
        end

        NVTX.@range "r$(i.max_radius) 2" begin
            gpu_out = ApplyFFTstencil(i, data, t_steps, ms2)
        end

        NVTX.@range "r$(i.max_radius) standard" begin
            gpu_out = ApplyStencil(i, data, 16)
        end
    end
end
bench(st_insts)
exit()
