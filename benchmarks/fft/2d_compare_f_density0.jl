include("../../module/CustomStencil.jl")
include("../../misc/misc.jl")
include("../../misc/cpu_stencils.jl")
## Star Stencil definition with radius = 4



##

dense_coefs = ones(9,9,9)

for i = parse(Int, ARGS[1])


    dense_coefs = zeros(9,9)
    dense_coefs[1:9,5] .= 1
    dense_coefs[5,1:9] .= 1
    d = (5-i:5+i)
    dense_coefs[d,d] .= 1
    @show sum(dense_coefs .==1)/729
    st_def = CreateStencilDefinition(dense_coefs)
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
