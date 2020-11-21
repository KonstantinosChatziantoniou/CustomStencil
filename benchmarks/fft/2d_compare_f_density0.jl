include("../../module/CustomStencil.jl")
include("../../misc/misc.jl")
include("../../misc/cpu_stencils.jl")
## Star Stencil definition with radius = 4



##

for i = parse(Int,ARGS[1])

    star_coefs = zeros(Float64, 9,9,9)
    star_coefs[1:9,5,5] .= 1
    star_coefs[5,1:9,5] .= 1

    d = 5-i:5+i
    star_coefs[d,d,5] .= 1
    @show sum(star_coefs .== 1)/81
    st_def = CreateStencilDefinition(star_coefs)
    bdim = 32
    if length(ARGS) == 2
        bdim = 16
    end
    st_inst1 = NewStencilInstance(st_def, bdim=bdim, m_step=false)

    radius = 4
    nx = 12
    ny = 12
    nz = 0
    dx = 1<<(nx)
    dy = 1<<(ny)
    dz = 1<<(nz)

    data = CreateData(dx,dy,dz)

    t_steps = 16
    gpu_out = ApplyStencil(st_inst1, data, t_steps)
    CUDA.cuProfilerStart()
    NVTX.@range "standard" begin
        gpu_out = ApplyStencil(st_inst1, data, t_steps) end
    exit()

end
